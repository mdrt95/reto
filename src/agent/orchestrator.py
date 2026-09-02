"""Bounded orchestration that keeps model output behind typed policies."""

import json
import logging
import re
from typing import Literal, Protocol, get_args

from src.agent.answer_planning import (
    AnswerPlanner,
    DirectAnswerRenderer,
    SynthesisFallbackRenderer,
    ToolResult,
    plan_trace_fields,
)
from src.agent.contracts import (
    MAX_DELIVERED_FACT_IDS,
    MAX_DISCUSSED_SOURCE_IDS,
    MAX_DISCUSSED_TOPICS,
    AnswerMode,
    AnswerPlan,
    AnswerTopic,
    AgentResponse,
    AgentTrace,
    ClaimKind,
    ConversationState,
    GeneratedResponse,
    GenerationUnavailableError,
    GroundingResult,
    Intent,
    IntentDecision,
    InvalidStructuredOutputError,
    SynthesisTransformation,
)
from src.agent.grounding import profile_source_ids, verify_claims
from src.agent.rephrase import (
    count_sentences,
    verify_rephrase,
    verify_synthesis_structure,
    verify_synthesis_text,
)
from src.guardrails.input_guard import evaluate_input
from src.guardrails.output_guard import evaluate_output
from src.models.profile import Profile
from src.tools.profile_tools import (
    ExperienceFilterResult,
    detect_resume_topic,
    ProfileQueryResult,
    ProfileSummaryPlan,
    ProjectSearchResult,
    ResumeFact,
    ResumeSearchResult,
    ResumeTopic,
    FilterExperienceArguments,
    QueryProfileArguments,
    SearchProjectsArguments,
    SearchResumeArguments,
    SummarizeProfileArguments,
    filter_experience,
    build_resume_fact_catalog,
    detect_response_language,
    detect_resume_topic,
    fact_display_text,
    find_unknown_entities,
    normalize_resume_text,
    query_profile,
    search_projects,
    search_resume,
    summarize_profile,
)

_ANSWER_TOPICS = frozenset(get_args(AnswerTopic))
"""The topics a trace may name; anything else is not a topic this state can record."""


def _referent_correction(language: Literal["en", "es"]) -> str:
    """State that an asserted antecedent was never delivered, before answering it."""
    return (
        "No he mencionado eso en esta conversación, pero sí está en el perfil de Marco:"
        if language == "es"
        else "I haven't mentioned that in this conversation, but it is in Marco's profile:"
    )


def _unnarrated_fact_fallback(fact: ResumeFact, language: Literal["en", "es"]) -> str:
    """Explain honestly why one profile index term cannot stand as a complete answer."""
    if language == "es":
        return (
            f'El perfil incluye "{fact.text}", pero ese elemento por sí solo no aporta '
            "una respuesta completa. Pregunta por la sección del perfil donde aparece."
        )
    return (
        f'The profile lists "{fact.text}", but that item alone does not provide a '
        "complete answer. Please ask about the profile section where it appears."
    )


def _bounded(values: list[str], limit: int) -> list[str]:
    """Deduplicate oldest-first and keep the most recent entries within the cap."""
    return list(dict.fromkeys(values))[-limit:]


_REFERENT_PHRASES = (
    "the part where it says", "the part where you said", "the part that says",
    "the bit where it says", "when you said", "where you said",
    "where you mentioned", "when you mentioned", "you said that", "you mentioned that",
    "la parte donde dice", "la parte donde dijiste", "la parte que dice",
    "cuando dijiste", "donde dijiste", "donde mencionaste", "cuando mencionaste",
    "dijiste que", "mencionaste que",
)
"""Phrases that assert a specific antecedent was already said.

Only phrases carrying their antecedent inline qualify. A bare referent ("that bit")
names nothing checkable and must stay on the clarification path.
"""

_RANKING_MARKERS = (
    "rank", "ranking", "best to worst", "worst to best",
    "ordena", "clasifica", "mejor a peor", "peor a mejor", "del mejor",
)

_SUMMARY_FIELD_MARKERS: dict[str, tuple[str, ...]] = {
    "skills": ("skill", "habilidad", "tecnolog", "stack"),
    "languages": ("idioma", "language"),
    "education": ("educa", "estudios", "degree"),
}

_GENERATION_LOGGER = logging.getLogger("banorte_cv_agent.generation")


_FEEDBACK_TEMPLATES = {
    "unsupported_vocabulary": (
        "The previous answer used wording that does not appear in the supplied facts: "
        "{details}. Rewrite it using only wording drawn from the facts, and list in "
        "`fact_ids` every fact each proposition takes wording from."
    ),
    "verb_drift": (
        "The previous answer changed what Marco did: {details}. Reuse each cited fact's "
        "own verb instead of a stronger, weaker, or different one."
    ),
    "too_long": (
        "The previous answer was over budget: {details}. Rewrite it shorter, keeping "
        "only the most representative evidence."
    ),
}
_DEFAULT_FEEDBACK = (
    "The previous answer was rejected ({code}: {details}). Rewrite it staying strictly "
    "inside the supplied facts and within every stated limit."
)


def _transformation_feedback(code: str, details: list[str]) -> str:
    """Turn one deterministic rejection into a correction the provider can act on."""
    joined = "; ".join(details) if details else "no further detail"
    template = _FEEDBACK_TEMPLATES.get(code)
    if template is not None:
        return template.format(details=joined)
    return _DEFAULT_FEEDBACK.format(code=code, details=joined)


def _fallback_reason_for(
    error: GenerationUnavailableError,
    *,
    stage: Literal["classifier", "generator", "rephraser"],
) -> str:
    """Derive a content-free fallback reason code from the caught exception.

    Never returns or logs the exception message itself (it may echo provider
    text) — only a fixed, small set of codes safe to log and trace.
    """
    if isinstance(error, InvalidStructuredOutputError):
        if "truncated" in str(error):
            return f"{stage}_truncated"
        return f"{stage}_invalid_output"
    return f"{stage}_unavailable"


def _log_generation_fallback(reason: str, *, stage: str) -> None:
    """Emit a content-free warning so a silent fallback is diagnosable in logs."""
    _GENERATION_LOGGER.warning(json.dumps({"event": "generation_fallback", "reason": reason, "stage": stage}))


_LIST_QUERY_HEADINGS: dict[str, dict[str, str]] = {
    "skills": {
        "en": "Programming languages and skills from the profile:",
        "es": "Lenguajes y habilidades del perfil:",
    },
    "languages": {
        "en": "Languages from the profile:",
        "es": "Idiomas del perfil:",
    },
    "education": {
        "en": "Education from the profile:",
        "es": "Educación del perfil:",
    },
    "current_role": {
        "en": "Current role from the profile:",
        "es": "Puesto actual según el perfil:",
    },
    "companies": {
        "en": "Employers from the profile:",
        "es": "Empleadores según el perfil:",
    },
}

_SKILL_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "programming_languages": {"en": "Programming languages", "es": "Lenguajes de programación"},
    "ai_llm": {"en": "AI / LLM", "es": "IA / LLM"},
    "ai_stack": {"en": "AI stack", "es": "Stack de IA"},
    "backend_apis": {"en": "Backend and APIs", "es": "Backend y APIs"},
    "devops_engineering": {"en": "DevOps and engineering", "es": "DevOps e ingeniería"},
}

_FALLBACK_NOTICE = {
    "en": (
        "I couldn't compose a written answer right now, so here are the verified "
        "profile facts instead:"
    ),
    "es": (
        "No pude redactar una respuesta en este momento, así que estos son los "
        "datos verificados del perfil:"
    ),
}

class IntentClassifier(Protocol):
    """Port for a model-backed structured intent classifier."""

    def classify(self, message: str, history: list[object]) -> IntentDecision: ...


class ResponseGenerator(Protocol):
    """Port for a model-backed response generator with explicit source scope."""

    def generate(
        self,
        *,
        message: str,
        history: list[object],
        allowed_facts: list[ResumeFact],
        tool_result: object | None,
        allowed_source_ids: set[str],
    ) -> GeneratedResponse: ...


class Rephraser(Protocol):
    """Port for a model-backed rewrite of already-selected facts, gated by verify_rephrase."""

    def rephrase(
        self,
        *,
        message: str,
        facts: list[ResumeFact],
        language: Literal["en", "es"],
    ) -> SynthesisTransformation: ...


class AgentService:
    """Run one bounded, traceable profile-answering workflow per chat turn."""

    def __init__(
        self,
        *,
        profile: Profile,
        classifier: IntentClassifier,
        generator: ResponseGenerator,
        rephraser: Rephraser | None = None,
    ) -> None:
        self._profile = profile
        self._classifier = classifier
        self._generator = generator
        self._rephraser = rephraser
        self._answer_planner = AnswerPlanner(profile)
        self._direct_answer_renderer = DirectAnswerRenderer(profile)
        self._synthesis_fallback_renderer = SynthesisFallbackRenderer(profile)

    def respond(
        self,
        message: str,
        *,
        history: list[object],
        state: ConversationState | None = None,
    ) -> AgentResponse:
        """Answer one turn and record which selection path produced the answer."""
        response = self._respond(message, history=history, state=state)
        if (
            response.trace.referent_source is None
            and response.trace.guardrail_input != "blocked"
            and response.trace.tool_name == "search_resume"
            and (
                bool(response.trace.selected_fact_ids)
                or response.trace.rendering_mode == "follow_up_exhausted"
            )
            and self._is_progressive_follow_up(message)
            and self._explicit_follow_up_unit(message) is not None
        ):
            response.trace.referent_source = "message"
        self._apply_informativeness_floor(response, message)
        if response.trace.selection_path is None and response.trace.guardrail_input != "blocked":
            # Recovery marks itself; everything else that reached fact selection is
            # ordinary success or an explicit empty selection, never folded together.
            response.trace.selection_path = (
                "primary" if response.trace.selected_fact_ids else "none"
            )
        if (
            response.trace.selected_fact_ids
            and self._referent_verdict(message, state) == "undelivered"
        ):
            # Correct the premise in front of the answer the ordinary path produced,
            # rather than rendering a second time here: the selection discipline, the
            # gates, and the rendering mode all stay exactly what they would be, and
            # only one honest sentence about this conversation is added.
            response.answer = (
                f"{_referent_correction(detect_response_language(message))}\n"
                f"{response.answer}"
            )
            response.trace.referent_correction = True
        if (
            response.trace.final_word_count is not None
            or response.trace.informativeness_outcome == "fallback"
        ):
            response.trace.final_word_count = len(response.answer.split())
        if (
            response.trace.final_sentence_count is not None
            or response.trace.informativeness_outcome == "fallback"
        ):
            response.trace.final_sentence_count = count_sentences(response.answer)
        response.state = self._accumulate_discourse(
            previous=state,
            current=response.state,
            trace=response.trace,
        )
        return response

    def _apply_informativeness_floor(
        self,
        response: AgentResponse,
        message: str,
    ) -> None:
        """Prevent a raw catalog index term from becoming an entire public answer."""
        response.trace.informativeness_outcome = "pass"
        visible_answer = re.sub(
            r"\[([^\]]+)\]\((?:https?://)[^)\s]+\)",
            r"\1",
            response.answer,
        )
        normalized_answer = normalize_resume_text(visible_answer)
        if not normalized_answer:
            return

        selected_fact_ids = set(response.trace.selected_fact_ids)
        for fact in build_resume_fact_catalog(self._profile):
            if (
                fact.fact_id in selected_fact_ids
                and fact.narrative_en is None
                and fact.narrative_es is None
                and normalized_answer == normalize_resume_text(fact.text)
            ):
                response.answer = _unnarrated_fact_fallback(
                    fact,
                    detect_response_language(message),
                )
                response.trace.informativeness_outcome = "fallback"
                response.trace.rendering_mode = "informativeness_fallback"
                response.trace.grounding_status = "fact_rendered"
                response.trace.answer_topic = fact.topic
                response.trace.evidence_topics = [fact.topic]
                response.trace.selected_fact_ids = [fact.fact_id]
                response.trace.selected_source_ids = [fact.source_id]
                response.trace.claim_source_ids = [fact.source_id]
                if response.state is not None:
                    response.state.last_topic = fact.topic
                    response.state.last_source_ids = [fact.source_id]
                    response.state.last_entities = [fact.entity] if fact.entity else []
                return

    def _accumulate_discourse(
        self,
        *,
        previous: ConversationState | None,
        current: ConversationState | None,
        trace: AgentTrace,
    ) -> ConversationState | None:
        """Carry the discourse record forward across turns, derived from this turn's trace.

        Merging here rather than inside each answer path is deliberate. Every route
        already reports what it selected, so the record follows from the trace alone,
        and a route added later cannot forget to maintain it.

        The two layers keep their own lifetimes: `current` supplies the single-turn
        snapshot exactly as before, and a turn that produced no snapshot resets those
        fields rather than leaving a stale referent behind for the next `tell me more`.
        """
        catalog = build_resume_fact_catalog(self._profile)
        known_facts = {fact.fact_id for fact in catalog}
        known_sources = {fact.source_id for fact in catalog}
        turn_facts = [
            fact_id for fact_id in trace.selected_fact_ids if fact_id in known_facts
        ]
        turn_sources = [
            source_id
            for source_id in trace.selected_source_ids
            if source_id in known_sources
        ]
        if previous is None and current is None and not turn_facts and not turn_sources:
            return None

        base = current if current is not None else ConversationState(
            response_language=previous.response_language if previous else "en",
        )
        # Client-carried state is untrusted input. Keep only identifiers the catalog
        # still defines, so a forged or stale record cannot claim a fact was delivered.
        delivered = [
            fact_id
            for fact_id in (previous.delivered_fact_ids if previous else [])
            if fact_id in known_facts
        ]
        discussed_sources = [
            source_id
            for source_id in (previous.discussed_source_ids if previous else [])
            if source_id in known_sources
        ]
        discussed_topics = list(previous.discussed_topics) if previous else []
        focus = previous.focus_source_id if previous else None
        if focus is not None and focus not in known_sources:
            focus = None

        if turn_facts or turn_sources:
            delivered = _bounded(delivered + turn_facts, MAX_DELIVERED_FACT_IDS)
            discussed_sources = _bounded(
                discussed_sources + turn_sources, MAX_DISCUSSED_SOURCE_IDS
            )
            for topic in trace.evidence_topics or [trace.answer_topic]:
                if topic in _ANSWER_TOPICS:
                    discussed_topics = _bounded(
                        [*discussed_topics, topic], MAX_DISCUSSED_TOPICS
                    )
            roots = list(
                dict.fromkeys(
                    source_id.split(".highlight:", 1)[0] for source_id in turn_sources
                )
            )
            # One unit answered from is a referent; several is an ambiguity, and
            # carrying the older focus through would answer about the wrong thing.
            focus = roots[0] if len(roots) == 1 else None

        # Revalidated rather than copied: the caps are the model's promise, and this
        # is the one writer, so a cap violation must fail here rather than ship.
        return ConversationState.model_validate(
            {
                **base.model_dump(),
                "focus_source_id": focus,
                "delivered_fact_ids": delivered,
                "discussed_topics": discussed_topics,
                "discussed_source_ids": discussed_sources,
            }
        )

    def _respond(
        self,
        message: str,
        *,
        history: list[object],
        state: ConversationState | None = None,
    ) -> AgentResponse:
        """Answer safely or return a verified boundary response for this turn."""
        input_result = evaluate_input(message)
        if not input_result.allowed:
            return AgentResponse(
                answer=input_result.message,
                trace=AgentTrace(guardrail_input="blocked"),
            )

        if self._referent_verdict(message, state) == "absent":
            return self._negative_referent_response(message, state)

        unknown_entities = find_unknown_entities(self._profile, message)
        if unknown_entities:
            language = detect_response_language(message)
            boundary_plan = self._answer_planner.boundary_plan(message, state)
            entities = ", ".join(unknown_entities)
            answer = (
                f"No encontré nada sobre {entities} en el perfil de Marco, "
                "así que no puedo opinar al respecto."
                if language == "es"
                else f"I couldn't find anything about {entities} in Marco's profile, "
                "so I can't comment on it."
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    grounding_status="profile_missing",
                    **plan_trace_fields(boundary_plan, "canonical_not_found"),
                ),
            )

        explicit_plan = self._answer_planner.explicit_direct_plan(message)
        follow_up = self._follow_up_plan(message, state, history)
        explicitly_named_unit = (
            self._explicit_follow_up_unit(message)
            if isinstance(follow_up, SearchResumeArguments)
            else None
        )
        if explicit_plan is not None and explicitly_named_unit is None:
            # A direct topic in the current message still outranks conversation
            # history. A named unit takes the progressive route so it can deepen.
            follow_up = None
        elif isinstance(follow_up, SearchResumeArguments):
            explicit_plan = None
        referent_source: Literal["message", "state"] | None = (
            "message" if explicit_plan is not None else None
        )
        if explicit_plan is None and follow_up is None:
            state_plan = self._answer_planner.explicit_direct_plan(message, state)
            if state_plan is not None:
                explicit_plan, referent_source = state_plan, "state"
        normalized_message = " ".join(message.casefold().split())
        if any(marker in normalized_message for marker in _RANKING_MARKERS):
            language = detect_response_language(message)
            boundary_plan = self._answer_planner.boundary_plan(message, state)
            answer = (
                "No puedo ordenar la experiencia de Marco de forma subjetiva. Indica un "
                "criterio objetivo del perfil, como una tecnología, etiqueta o rol, y "
                "filtraré por él."
                if language == "es"
                else "I can't rank Marco's experience subjectively. Tell me an objective "
                "criterion from the profile, such as a technology, tag, or role, and "
                "I'll filter by it."
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    grounding_status="clarification",
                    **plan_trace_fields(boundary_plan, "clarification"),
                ),
            )

        fallback_reason: str | None = None
        try:
            decision = self._classifier.classify(message, history)
        except GenerationUnavailableError as error:
            fallback_reason = _fallback_reason_for(error, stage="classifier")
            _log_generation_fallback(fallback_reason, stage="classifier")
            decision = self._bounded_intent_fallback(message)
            if decision is None and explicit_plan is not None:
                decision = IntentDecision(intent=Intent.DIRECT_QUESTION, confidence=1.0)
            if decision is None:
                if follow_up is None and self._resume_search_arguments(message, state) is None:
                    # The classifier only ever chose a tool; it never chose the facts.
                    # An anchor evidenced in the message is enough to answer without it,
                    # and anything less deflects at HTTP 200 like every other boundary.
                    topic = detect_resume_topic(self._profile, message)
                    if topic is None:
                        return self._unclassified_response(message, state, fallback_reason)
                    follow_up = SearchResumeArguments(query=message, topic=topic)
                decision = IntentDecision(intent=Intent.FOLLOW_UP, confidence=1.0)
        if follow_up == "clarify":
            language = detect_response_language(message)
            boundary_plan = self._answer_planner.boundary_plan(message, state)
            answer = (
                "¿A qué parte o elemento anterior te refieres?"
                if language == "es"
                else "Which part or item from the previous answer do you mean?"
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                    grounding_status="clarification",
                    fallback_reason=fallback_reason,
                    **plan_trace_fields(boundary_plan, "clarification"),
                ),
                state=state,
            )
        if isinstance(follow_up, SearchResumeArguments):
            # A verified focused plan is stronger evidence than the classifier's
            # coarse intent. Normalize the trace decision before any classifier-
            # dependent boundary can divert the turn.
            decision = IntentDecision(intent=Intent.FOLLOW_UP, confidence=1.0)
        if explicit_plan is not None:
            explicit_response = self._direct_plan_response(
                plan=explicit_plan,
                decision=decision,
                message=message,
                fallback_reason=fallback_reason,
            )
            explicit_response.trace.referent_source = referent_source
            return explicit_response
        if decision.intent in {Intent.OUT_OF_SCOPE, Intent.ADVERSARIAL}:
            language = detect_response_language(message)
            answer = (
                "Me enfoco en el perfil profesional de Marco, su experiencia, habilidades "
                "y proyectos. ¿Qué te gustaría saber?"
                if language == "es"
                else "I'm focused on Marco's professional profile, experience, skills, and "
                "projects. What would you like to know?"
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    guardrail_input="blocked",
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                    fallback_reason=fallback_reason,
                ),
            )
        if decision.confidence < 0.7 and self._resume_search_arguments(message, state) is None:
            language = detect_response_language(message)
            boundary_plan = self._answer_planner.boundary_plan(message, state)
            answer = (
                "¿Podrías aclarar a qué parte del perfil profesional de Marco te refieres?"
                if language == "es"
                else "Could you clarify which part of Marco's professional profile you mean?"
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                    grounding_status="clarification",
                    fallback_reason=fallback_reason,
                    **plan_trace_fields(boundary_plan, "clarification"),
                ),
            )

        tool_name, tool_result = self._execute_tool(decision, message)
        if isinstance(follow_up, SearchResumeArguments):
            tool_name, tool_result = "search_resume", search_resume(self._profile, follow_up)
        elif not self._tool_has_facts(tool_result):
            universal_arguments = self._resume_search_arguments(message, state)
            if universal_arguments is not None:
                tool_name = "search_resume"
                tool_result = search_resume(self._profile, universal_arguments)
        answer_plan = self._answer_planner.plan_from_tool(
            message,
            tool_result,
        )
        if (
            isinstance(follow_up, SearchResumeArguments)
            and follow_up.source_ids
            and follow_up.limit == 1
            and isinstance(tool_result, ResumeSearchResult)
        ):
            # Generic deepening is selection, not transformation. Some of its phrase
            # family predates this route as a broad "conclusion" marker, so pin the
            # plan back to the one fact the focused search actually selected.
            answer_plan = answer_plan.model_copy(
                update={
                    "mode": AnswerMode.DIRECT,
                    "synthesis_dimension": None,
                    "selected_fact_ids": [fact.fact_id for fact in tool_result.matches],
                    "selected_source_ids": [fact.source_id for fact in tool_result.matches],
                }
            )
        if isinstance(tool_result, ProfileQueryResult):
            tool_result_count = len(tool_result.value)
        elif isinstance(tool_result, ProfileSummaryPlan):
            tool_result_count = len(tool_result.source_ids)
        else:
            tool_result_count = len(getattr(tool_result, "matches", [])) if tool_result else 0
        if isinstance(tool_result, ResumeSearchResult) and tool_result.profile_missing:
            if (
                isinstance(follow_up, SearchResumeArguments)
                and follow_up.source_ids
                and follow_up.exclude_fact_ids
                and self._focused_unit_is_exhausted(follow_up)
            ):
                return self._follow_up_exhausted_response(
                    decision=decision,
                    arguments=follow_up,
                    answer_plan=answer_plan,
                    state=state,
                )
            return self._profile_missing_response(
                decision, tool_result, tool_name, answer_plan
            )
        if isinstance(tool_result, ProfileQueryResult) and tool_result.field in {
            "skills",
            "languages",
            "education",
            "current_role",
            "companies",
        } and answer_plan.synthesis_dimension is None:
            return self._list_rendered_response(
                decision=decision,
                tool_name=tool_name,
                tool_result=tool_result,
                tool_result_count=tool_result_count,
                message=message,
                answer_plan=answer_plan,
            )
        if answer_plan.mode is AnswerMode.DIRECT and answer_plan.selected_fact_ids:
            return self._direct_plan_response(
                plan=answer_plan,
                decision=decision,
                message=message,
                fallback_reason=fallback_reason,
                tool_name_override=tool_name,
                tool_result_count=tool_result_count,
            )
        if answer_plan.synthesis_dimension is not None:
            synthesis_response = self._bounded_synthesis_response(
                decision=decision,
                tool_name=tool_name,
                tool_result=tool_result,
                tool_result_count=tool_result_count,
                message=message,
                history=history,
                answer_plan=answer_plan,
                initial_fallback_reason=fallback_reason,
            )
            if synthesis_response is not None:
                return synthesis_response
        allowed_sources = profile_source_ids(self._profile)
        tool_ordered_fact_ids = self._answer_planner.ordered_fact_ids(tool_result)
        if not tool_ordered_fact_ids:
            # Zero selection is an explicit state, not a degenerate success. Asking the
            # generator for grounded claims with no allowed facts is unanswerable by
            # construction and surfaces as a 503.
            return self._zero_selection_response(
                decision=decision,
                message=message,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                fallback_reason=fallback_reason,
                state=state,
            )
        selected_fact_ids = set(tool_ordered_fact_ids)
        allowed_facts = self._allowed_facts(selected_fact_ids)
        if self._rephraser is not None:
            if len(tool_ordered_fact_ids) <= 8:
                rendered = self._fact_selection_response(
                    decision=decision,
                    ordered_fact_ids=tool_ordered_fact_ids,
                    tool_name=tool_name,
                    tool_result=tool_result,
                    tool_result_count=tool_result_count,
                    message=message,
                    generator_skipped=True,
                    answer_plan=answer_plan,
                    fallback_reason=fallback_reason,
                )
                if rendered is not None:
                    return rendered
        try:
            generated = self._generator.generate(
                message=message,
                history=history,
                allowed_facts=allowed_facts,
                tool_result=tool_result,
                allowed_source_ids=allowed_sources,
            )
        except GenerationUnavailableError as error:
            fallback_reason = _fallback_reason_for(error, stage="generator")
            _log_generation_fallback(fallback_reason, stage="generator")
            fallback = self._tool_fallback_response(
                decision=decision,
                tool_name=tool_name,
                tool_result=tool_result,
                tool_result_count=tool_result_count,
                message=message,
                fallback_reason=fallback_reason,
                answer_plan=answer_plan,
            )
            if fallback is not None:
                return fallback
            raise
        grounding = verify_claims(
            self._profile,
            generated.claims,
            selected_fact_ids=selected_fact_ids,
        )
        if grounding.claim_fact_ids:
            ordered_fact_ids = list(
                dict.fromkeys(
                    fact_id
                    for claim_index in sorted(grounding.claim_fact_ids)
                    for fact_id in grounding.claim_fact_ids[claim_index]
                )
            )
            rendered = self._fact_selection_response(
                decision=decision,
                ordered_fact_ids=ordered_fact_ids,
                tool_name=tool_name,
                tool_result=tool_result,
                tool_result_count=tool_result_count,
                message=message,
                answer_plan=answer_plan,
                fallback_reason=fallback_reason,
            )
            if rendered is not None:
                return rendered
        if grounding.status != "fully_grounded":
            try:
                generated = self._generator.generate(
                    message=message,
                    history=history,
                    allowed_facts=allowed_facts,
                    tool_result=tool_result,
                    allowed_source_ids=allowed_sources,
                )
            except GenerationUnavailableError as error:
                fallback_reason = _fallback_reason_for(error, stage="generator")
                _log_generation_fallback(fallback_reason, stage="generator")
                fallback = self._tool_fallback_response(
                    decision=decision,
                    tool_name=tool_name,
                    tool_result=tool_result,
                    tool_result_count=tool_result_count,
                    message=message,
                    fallback_reason=fallback_reason,
                    answer_plan=answer_plan,
                )
                if fallback is not None:
                    return fallback
                raise
            grounding = verify_claims(
                self._profile,
                generated.claims,
                selected_fact_ids=selected_fact_ids,
            )
        accepted_claim_indexes = list(grounding.claim_sources)
        accepted_source_ids = [
            source_id
            for index in accepted_claim_indexes
            for source_id in grounding.claim_sources[index]
        ]
        # Deliver only text that crossed the typed claim-verification boundary.
        # Provider prose outside the claims array is untrusted and intentionally dropped.
        answer = "\n\n".join(
            generated.claims[index].text for index in accepted_claim_indexes
        )
        if grounding.status != "fully_grounded":
            accepted_claim_indexes = [
                index
                for index in accepted_claim_indexes
                if generated.claims[index].kind is ClaimKind.DIRECT
            ]
            verified_facts = list(
                dict.fromkeys(generated.claims[index].text for index in accepted_claim_indexes)
            )
            accepted_source_ids = [
                source_id
                for index in accepted_claim_indexes
                for source_id in grounding.claim_sources[index]
            ]
            used_tool_fallback = False
            if not verified_facts:
                verified_facts, accepted_source_ids = self._verified_tool_facts(
                    tool_result, detect_response_language(message)
                )
                used_tool_fallback = True
            if not verified_facts:
                language = detect_response_language(message)
                answer = (
                    "Solo puedo confirmar información explícitamente respaldada por el "
                    "perfil de Marco."
                    if language == "es"
                    else "I can only confirm information that is explicitly supported by "
                    "Marco's profile."
                )
                return AgentResponse(
                    answer=answer,
                    trace=AgentTrace(
                        intent=decision.intent.value,
                        intent_confidence=decision.confidence,
                        tool_name=tool_name,
                        tool_result_count=tool_result_count,
                        grounding_status=grounding.status,
                        fallback_reason=fallback_reason,
                        **plan_trace_fields(answer_plan, "canonical_fallback"),
                    ),
                )
            if used_tool_fallback and isinstance(tool_result, ProfileSummaryPlan):
                body = "\n".join(f"- {fact}" for fact in verified_facts)
            else:
                body = "\n\n".join(verified_facts)
            answer = body
            if used_tool_fallback:
                language = detect_response_language(message)
                answer = f"{_FALLBACK_NOTICE[language]}\n{body}"

        output_result = evaluate_output(answer, self._profile)
        if not output_result.allowed:
            language = detect_response_language(message)
            answer = (
                "Puedo ayudarte con el perfil profesional público de Marco, pero no "
                "puedo proporcionar esa información."
                if language == "es"
                else "I can help with Marco's public professional profile, but I can't "
                "provide that information."
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    tool_name=tool_name,
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                    tool_result_count=tool_result_count,
                    grounding_status=grounding.status,
                    guardrail_output="blocked",
                    fallback_reason=fallback_reason,
                    **plan_trace_fields(answer_plan, "canonical_fallback"),
                ),
            )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status=grounding.status,
                claim_source_ids=accepted_source_ids,
                fallback_reason=fallback_reason,
                **plan_trace_fields(answer_plan, "generated"),
            ),
            state=self._state_from_result(tool_name, tool_result, accepted_source_ids, message),
        )

    def _direct_plan_response(
        self,
        *,
        plan: AnswerPlan,
        decision: IntentDecision,
        message: str,
        fallback_reason: str | None,
        tool_name_override: str | None = None,
        tool_result_count: int | None = None,
    ) -> AgentResponse:
        """Render a selected direct plan locally, with no generation or rephrase call."""
        catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(self._profile)}
        selected = [catalog[fact_id] for fact_id in plan.selected_fact_ids]
        answer = self._direct_answer_renderer.render(plan)
        output_result = evaluate_output(answer, self._profile)
        if not output_result.allowed:
            answer = (
                "Puedo ayudarte con el perfil profesional público de Marco, pero no puedo proporcionar esa información."
                if plan.language == "es"
                else "I can help with Marco's public professional profile, but I can't provide that information."
            )
        tool_name = tool_name_override or {
            "tag": "filter_experience",
            "projects": "search_projects",
        }.get(plan.requested_field, "search_resume")
        result = ResumeSearchResult(
            query=message,
            language=plan.language,
            topic=plan.topic,
            matches=selected,
        )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count if tool_result_count is not None else len(selected),
                grounding_status="fact_rendered",
                guardrail_output="pass" if output_result.allowed else "blocked",
                claim_source_ids=plan.selected_source_ids,
                fallback_reason=fallback_reason,
                generator_skipped=True,
                answer_mode=plan.mode.value,
                rendering_mode="canonical",
                answer_topic=plan.topic,
                evidence_topics=plan.evidence_topics or [plan.topic],
                answer_scope=plan.scope,
                requested_field=plan.requested_field,
                selected_fact_ids=plan.selected_fact_ids,
                selected_source_ids=plan.selected_source_ids,
            ),
            state=self._state_from_result(tool_name, result, plan.selected_source_ids, message),
        )

    def _execute_tool(
        self,
        decision: IntentDecision,
        message: str,
    ) -> tuple[str | None, ToolResult | None]:
        """Translate only validated intent fields into allowlisted read-only tools."""
        has_filter_plan = (
            decision.filter_by in {"technology", "tag", "role"}
            and bool(decision.filter_value)
        )
        if decision.intent is Intent.FILTER_REQUEST or (
            decision.intent is Intent.SEARCH_QUERY and has_filter_plan
        ):
            if self._is_explicit_project_question(message):
                fallback_terms = self._project_fallback_terms(message)
                project_query = decision.query or decision.filter_value or fallback_terms[0]
                project_result = self._search_projects_with_fallback(project_query, message)
                if project_result.matches:
                    return "search_projects", project_result
            filter_by = decision.filter_by if decision.filter_by in {"technology", "tag", "role"} else "tag"
            filter_value = decision.filter_value
            if not has_filter_plan:
                tag_override = self._answer_planner.profile_tag_match(message)
                if tag_override is not None:
                    filter_by = "tag"
                    filter_value = tag_override
            result = filter_experience(
                self._profile,
                FilterExperienceArguments(filter_by=filter_by, value=filter_value or "profile"),
            )
            if not result.matches and self._is_explicit_project_question(message):
                return "search_projects", self._search_projects_with_fallback(
                    decision.query or decision.filter_value or "profile",
                    message,
                )
            return "filter_experience", result
        if decision.intent is Intent.SEARCH_QUERY:
            return "search_projects", self._search_projects_with_fallback(
                decision.query or "profile",
                message,
            )
        if decision.intent is Intent.SUMMARY_REQUEST:
            tag_override = self._answer_planner.profile_tag_match(message)
            if tag_override is not None:
                return "filter_experience", filter_experience(
                    self._profile,
                    FilterExperienceArguments(filter_by="tag", value=tag_override),
                )
            field_override = self._summary_field_override(message)
            if field_override is not None:
                return "query_profile", query_profile(
                    self._profile,
                    QueryProfileArguments(field=field_override),
                )
            audience = decision.audience if decision.audience in {"technical", "recruiter", "executive"} else "recruiter"
            return "summarize_profile", summarize_profile(
                self._profile,
                SummarizeProfileArguments(audience=audience),
            )
        profile_field = decision.profile_field
        explicit_field = self._explicit_profile_field(message)
        if explicit_field is not None:
            profile_field = explicit_field
        elif profile_field == "companies" and not self._is_employment_history_question(message):
            # Employer projection is intentionally narrow: a coarse classifier field
            # cannot turn an unrelated direct question into an employer list.
            profile_field = None
        if (
            decision.intent in {Intent.DIRECT_QUESTION, Intent.FOLLOW_UP}
            and profile_field is None
            and self._is_employment_history_question(message)
        ):
            profile_field = "companies"
        if (
            decision.intent in {Intent.DIRECT_QUESTION, Intent.FOLLOW_UP}
            and profile_field in (None, "skills")
            and self._mentions_project_or_experience_technology(message)
        ):
            technology_result = self._technology_search_result(message)
            if technology_result is not None:
                return "search_resume", technology_result
        if decision.intent in {Intent.DIRECT_QUESTION, Intent.FOLLOW_UP} and profile_field in {
            "skills",
            "languages",
            "education",
            "current_role",
            "companies",
        }:
            return "query_profile", query_profile(
                self._profile,
                QueryProfileArguments(field=profile_field),
            )
        return None, None

    def _mentions_project_or_experience_technology(self, message: str) -> bool:
        """True when the message names a technology or tag cataloged under a project or job.

        Built entirely from `build_resume_fact_catalog` (D-020) — no hardcoded technology
        list — so a question naming one specific technology (e.g. "FAISS") is recognized
        without special-casing it, keeping the heuristic profile-derived and bounded.
        """
        catalog = build_resume_fact_catalog(self._profile)
        technology_tokens: set[str] = set()
        for fact in catalog:
            if fact.topic not in ("projects", "experience"):
                continue
            for keyword in fact.keywords:
                technology_tokens.update(normalize_resume_text(keyword).split())
        message_tokens = set(normalize_resume_text(message).split())
        return bool(technology_tokens & message_tokens)

    def _technology_search_result(self, message: str) -> ResumeSearchResult | None:
        """Search facts for one named technology, preferring the profile's own topic guess.

        Falls back to the project and experience domains explicitly (D-032) because the
        generic topic detector in `search_resume` can pick an unrelated topic — e.g. one
        matching the verb "worked" — before ever inspecting the technology token itself.
        """
        for topic in (None, "projects", "experience"):
            result = search_resume(self._profile, SearchResumeArguments(query=message, topic=topic))
            if result.matches:
                return result
        return None

    def _verify_transformation(
        self,
        *,
        candidate: SynthesisTransformation,
        answer_plan: AnswerPlan,
        catalog: dict[str, ResumeFact],
        selected_facts: list[ResumeFact],
        message: str,
    ) -> tuple[str, str, list[str]]:
        """Return one transformation's verdict code, delivery text, and rejection detail.

        Checks run narrowest first: fact mapping, then each proposition against the facts
        it cites, then the assembled answer's structure and whole-answer containment.
        """
        selected_ids = set(answer_plan.selected_fact_ids)
        mapped_text = " ".join(
            proposition.text.strip() for proposition in candidate.propositions
        )
        mappings_are_valid = all(
            proposition.fact_ids and set(proposition.fact_ids) <= selected_ids
            for proposition in candidate.propositions
        )
        if not mappings_are_valid:
            return "missing_fact_ids", mapped_text, []
        assert answer_plan.synthesis_dimension is not None
        for proposition in candidate.propositions:
            verdict = verify_synthesis_text(
                text=proposition.text,
                selected_facts=[catalog[fact_id] for fact_id in proposition.fact_ids],
                vocabulary_facts=selected_facts,
                catalog=list(catalog.values()),
                language=answer_plan.language,
                dimension=answer_plan.synthesis_dimension,
            )
            if not verdict.allowed:
                return verdict.code, mapped_text, verdict.details
        structure_verdict = verify_synthesis_structure(
            text=mapped_text,
            proposition_fact_ids=[
                proposition.fact_ids for proposition in candidate.propositions
            ],
            proposition_texts=[
                proposition.text for proposition in candidate.propositions
            ],
            dimension=answer_plan.synthesis_dimension,
            detail_requested=self._answer_planner.detail_requested(message),
        )
        if not structure_verdict.allowed:
            return structure_verdict.code, mapped_text, structure_verdict.details
        whole_verdict = verify_synthesis_text(
            text=mapped_text,
            selected_facts=selected_facts,
            catalog=list(catalog.values()),
            language=answer_plan.language,
            dimension=answer_plan.synthesis_dimension,
        )
        if not whole_verdict.allowed:
            return whole_verdict.code, mapped_text, whole_verdict.details
        return "accepted", mapped_text, []

    def _bounded_synthesis_response(
        self,
        *,
        decision: IntentDecision,
        tool_name: str | None,
        tool_result: ToolResult | None,
        tool_result_count: int,
        message: str,
        history: list[object],
        answer_plan: AnswerPlan,
        initial_fallback_reason: str | None,
    ) -> AgentResponse | None:
        """Transform only the planner's bounded facts, or return one concise fallback."""
        assert answer_plan.synthesis_dimension is not None
        catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(self._profile)}
        selected_facts = [
            catalog[fact_id]
            for fact_id in answer_plan.selected_fact_ids
            if fact_id in catalog
        ]
        if not selected_facts:
            return None

        transformation_outcome: str
        fallback_reason = initial_fallback_reason
        rephrase_outcome: str | None = None
        answer: str | None = None
        generator_skipped = self._rephraser is not None
        impact_without_outcome = (
            answer_plan.synthesis_dimension == "impact"
            and not any(
                self._answer_planner.has_explicit_outcome(fact)
                for fact in selected_facts
            )
        )
        if impact_without_outcome:
            transformation_outcome = "rejected:missing_impact_evidence"
            fallback_reason = fallback_reason or "missing_explicit_impact"
            rephrase_outcome = "rejected:missing_impact_evidence"
            generator_skipped = True
        elif self._rephraser is not None:
            feedback: str | None = None
            mapped_text = ""
            verdict_code = "accepted"
            for attempt in range(2):
                try:
                    candidate = self._rephraser.rephrase(
                        message=message,
                        facts=selected_facts,
                        language=answer_plan.language,
                        # Omitted on the first attempt so a provider adapter without a
                        # corrective parameter still satisfies the interface.
                        **({"feedback": feedback} if feedback else {}),
                    )
                except GenerationUnavailableError as error:
                    reason = _fallback_reason_for(error, stage="rephraser")
                    _log_generation_fallback(reason, stage="rephraser")
                    fallback_reason = fallback_reason or reason
                    rephrase_outcome = reason
                    transformation_outcome = f"unavailable:{reason}"
                    break
                verdict_code, mapped_text, details = self._verify_transformation(
                    candidate=candidate,
                    answer_plan=answer_plan,
                    catalog=catalog,
                    selected_facts=selected_facts,
                    message=message,
                )
                if verdict_code == "accepted":
                    answer = mapped_text
                    rephrase_outcome = "accepted"
                    transformation_outcome = (
                        "accepted" if attempt == 0 else "accepted_after_correction"
                    )
                    break
                # The gate already names what was wrong, so one corrective attempt is
                # worth more than loosening a check. A provider that cannot use that
                # feedback will not succeed by plain repetition either, so never more.
                feedback = _transformation_feedback(verdict_code, details)
            else:
                rephrase_outcome = f"rejected:{verdict_code}"
                fallback_reason = fallback_reason or f"synthesis_rejected:{verdict_code}"
                transformation_outcome = rephrase_outcome
        else:
            try:
                generated = self._generator.generate(
                    message=message,
                    history=history,
                    allowed_facts=selected_facts,
                    tool_result=self._project_tool_result(
                        tool_result=tool_result,
                        selected_facts=selected_facts,
                        message=message,
                        language=answer_plan.language,
                        topic=answer_plan.topic,
                    ),
                    allowed_source_ids=set(answer_plan.selected_source_ids),
                )
            except GenerationUnavailableError as error:
                reason = _fallback_reason_for(error, stage="generator")
                _log_generation_fallback(reason, stage="generator")
                fallback_reason = fallback_reason or reason
                transformation_outcome = f"unavailable:{reason}"
            else:
                selected_ids = set(answer_plan.selected_fact_ids)
                mapped_claims = [
                    claim
                    for claim in generated.claims
                    if claim.fact_ids
                    and set(claim.fact_ids) <= selected_ids
                    and set(claim.source_ids) <= set(answer_plan.selected_source_ids)
                    and all(
                        catalog[fact_id].source_id in claim.source_ids
                        for fact_id in claim.fact_ids
                    )
                ]
                if len(mapped_claims) != len(generated.claims):
                    verdict_code = "missing_fact_ids"
                    transformation_outcome = f"rejected:{verdict_code}"
                    fallback_reason = fallback_reason or f"synthesis_rejected:{verdict_code}"
                else:
                    candidate = " ".join(claim.text.strip() for claim in mapped_claims)
                    proposition_verdicts = [
                        verify_synthesis_text(
                            text=claim.text,
                            selected_facts=[catalog[fact_id] for fact_id in claim.fact_ids],
                            vocabulary_facts=selected_facts,
                            catalog=list(catalog.values()),
                            language=answer_plan.language,
                            dimension=answer_plan.synthesis_dimension,
                        )
                        for claim in mapped_claims
                    ]
                    rejected = next(
                        (verdict for verdict in proposition_verdicts if not verdict.allowed),
                        None,
                    )
                    structure_verdict = verify_synthesis_structure(
                        text=candidate,
                        proposition_fact_ids=[claim.fact_ids for claim in mapped_claims],
                        proposition_texts=[claim.text for claim in mapped_claims],
                        dimension=answer_plan.synthesis_dimension,
                        detail_requested=self._answer_planner.detail_requested(message),
                    )
                    whole_verdict = verify_synthesis_text(
                        text=candidate,
                        selected_facts=selected_facts,
                        catalog=list(catalog.values()),
                        language=answer_plan.language,
                        dimension=answer_plan.synthesis_dimension,
                    )
                    verdict = rejected or (
                        structure_verdict if not structure_verdict.allowed else whole_verdict
                    )
                    if verdict.allowed:
                        answer = candidate
                        transformation_outcome = "accepted"
                    else:
                        fallback_reason = fallback_reason or f"synthesis_rejected:{verdict.code}"
                        transformation_outcome = f"rejected:{verdict.code}"

        rendering_mode = "transformed"
        grounding_status = "rephrased" if self._rephraser is not None else "fully_grounded"
        if answer is None:
            rendering_mode = "canonical_fallback"
            grounding_status = (
                "fact_rendered"
                if transformation_outcome.startswith("rejected") and self._rephraser is not None
                else "tool_fallback"
            )
            # No fallback notice here (D-024, amended): the other two fallback paths
            # emit a bullet list of facts that a reader could mistake for a composed
            # answer, but this one renders a single human-reviewed narrative that is
            # already prose. Announcing it degrades a correct answer, and the trace
            # still records `canonical_fallback` for anyone who needs the distinction.
            body = self._synthesis_fallback_renderer.render(answer_plan)
            if not body:
                return None
            answer = body

        output_result = evaluate_output(answer, self._profile)
        if not output_result.allowed:
            return None
        result = ResumeSearchResult(
            query=message,
            language=answer_plan.language,
            topic=answer_plan.topic,
            matches=selected_facts,
        )
        source_ids = list(dict.fromkeys(fact.source_id for fact in selected_facts))
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status=grounding_status,
                claim_source_ids=source_ids,
                rephrase_outcome=rephrase_outcome,
                fallback_reason=fallback_reason,
                generator_skipped=generator_skipped,
                transformation_outcome=transformation_outcome,
                final_word_count=len(answer.split()),
                final_sentence_count=count_sentences(answer),
                **plan_trace_fields(answer_plan, rendering_mode),
            ),
            state=self._state_from_result(tool_name, result, source_ids, message),
        )

    @staticmethod
    def _project_tool_result(
        *,
        tool_result: ToolResult | None,
        selected_facts: list[ResumeFact],
        message: str,
        language: Literal["en", "es"],
        topic: ResumeTopic,
    ) -> ToolResult:
        """Preserve the tool contract while removing every unselected fact."""
        fact_ids = {fact.fact_id for fact in selected_facts}
        source_ids = {fact.source_id for fact in selected_facts}
        if isinstance(tool_result, ResumeSearchResult):
            return tool_result.model_copy(
                update={"matches": [fact for fact in tool_result.matches if fact.fact_id in fact_ids]}
            )
        if isinstance(tool_result, ProjectSearchResult):
            return tool_result.model_copy(
                update={
                    "matches": [
                        match for match in tool_result.matches if match.source_id in source_ids
                    ]
                }
            )
        if isinstance(tool_result, ExperienceFilterResult):
            return tool_result.model_copy(
                update={
                    "matches": [
                        match for match in tool_result.matches if match.source_id in source_ids
                    ]
                }
            )
        if isinstance(tool_result, ProfileSummaryPlan):
            return tool_result.model_copy(
                update={
                    "fact_ids": [fact.fact_id for fact in selected_facts],
                    "source_ids": list(dict.fromkeys(fact.source_id for fact in selected_facts)),
                }
            )
        if isinstance(tool_result, ProfileQueryResult):
            selected_text = {fact.text for fact in selected_facts}
            return tool_result.model_copy(
                update={
                    "value": [value for value in tool_result.value if value in selected_text],
                    "source_ids": list(dict.fromkeys(fact.source_id for fact in selected_facts)),
                }
            )
        return ResumeSearchResult(
            query=message,
            language=language,
            topic=topic,
            matches=selected_facts,
        )

    def _tool_fallback_response(
        self,
        *,
        decision: IntentDecision,
        tool_name: str | None,
        tool_result: ToolResult | None,
        tool_result_count: int,
        message: str,
        fallback_reason: str | None = None,
        answer_plan: AnswerPlan,
    ) -> AgentResponse | None:
        """Return verified deterministic facts after model generation is unavailable."""
        language = detect_response_language(message)
        verified_facts, accepted_source_ids = self._verified_tool_facts(tool_result, language)
        if not verified_facts:
            return None
        notice = _FALLBACK_NOTICE[language]
        if isinstance(tool_result, ResumeSearchResult):
            body = self._render_resume_result(tool_result)
        elif isinstance(tool_result, ProfileSummaryPlan):
            body = "\n".join(f"- {fact}" for fact in verified_facts)
        else:
            body = "\n\n".join(verified_facts)
        answer = f"{notice}\n{body}"
        output_result = evaluate_output(answer, self._profile)
        if not output_result.allowed:
            return None
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status="tool_fallback",
                claim_source_ids=accepted_source_ids,
                fallback_reason=fallback_reason,
                **plan_trace_fields(answer_plan, "canonical_fallback"),
            ),
            state=self._state_from_result(tool_name, tool_result, accepted_source_ids, message),
        )

    def _fact_selection_response(
        self,
        *,
        decision: IntentDecision,
        ordered_fact_ids: list[str],
        tool_name: str | None,
        tool_result: ToolResult | None,
        tool_result_count: int,
        message: str,
        generator_skipped: bool = False,
        answer_plan: AnswerPlan,
        fallback_reason: str | None = None,
    ) -> AgentResponse | None:
        """Render provider- or tool-selected fact IDs exclusively from canonical fact values."""
        catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(self._profile)}
        selected_facts = [catalog[fact_id] for fact_id in ordered_fact_ids if fact_id in catalog]
        if not selected_facts:
            return None
        if isinstance(tool_result, ResumeSearchResult):
            topic = tool_result.topic
        else:
            topic = selected_facts[0].topic
        result = ResumeSearchResult(
            query=message,
            language=detect_response_language(message),
            topic=topic,
            matches=selected_facts,
        )
        answer = self._render_resume_result(result)
        grounding_status = "fact_rendered"
        rephrase_outcome: str | None = None
        rendering_fallback_reason = fallback_reason
        if self._rephraser is not None:
            try:
                transformation = self._rephraser.rephrase(
                    message=message,
                    facts=selected_facts,
                    language=result.language,
                )
            except GenerationUnavailableError as error:
                rephrase_outcome = _fallback_reason_for(error, stage="rephraser")
                rendering_fallback_reason = rephrase_outcome
                _log_generation_fallback(rephrase_outcome, stage="rephraser")
            else:
                rephrased_text = transformation.text
                verdict = verify_rephrase(
                    text=rephrased_text,
                    selected_facts=selected_facts,
                    catalog=list(catalog.values()),
                    language=result.language,
                )
                if verdict.allowed:
                    answer = rephrased_text
                    grounding_status = "rephrased"
                    rephrase_outcome = "accepted"
                else:
                    rephrase_outcome = f"rejected:{verdict.code}"
        output_result = evaluate_output(answer, self._profile)
        if not output_result.allowed:
            return None
        source_ids = list(dict.fromkeys(fact.source_id for fact in selected_facts))
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status=grounding_status,
                claim_source_ids=source_ids,
                rephrase_outcome=rephrase_outcome,
                fallback_reason=rendering_fallback_reason,
                generator_skipped=generator_skipped,
                **plan_trace_fields(
                    answer_plan,
                    "rephrased" if rephrase_outcome == "accepted" else "canonical",
                ),
            ),
            state=self._state_from_result(tool_name, result, source_ids, message),
        )

    def _search_projects_with_fallback(
        self,
        query: str,
        message: str,
    ) -> ProjectSearchResult:
        """Search a validated query, then bounded concepts if its full text is too broad."""
        result = search_projects(
            self._profile,
            SearchProjectsArguments(query=query),
        )
        if result.matches:
            return result
        matches = [
            match
            for term in self._project_fallback_terms(message)
            for match in search_projects(
                self._profile,
                SearchProjectsArguments(query=term),
            ).matches
        ]
        return ProjectSearchResult(
            matches=list({match.source_id: match for match in matches}.values())
        )

    def _verified_tool_facts(
        self,
        tool_result: ToolResult | None,
        language: Literal["en", "es"] = "en",
    ) -> tuple[list[str], list[str]]:
        """Return exact source-backed summaries when model grounding yields no usable claim."""
        if isinstance(tool_result, ExperienceFilterResult):
            facts = list(dict.fromkeys(match.summary for match in tool_result.matches))
            source_ids = list(dict.fromkeys(match.source_id for match in tool_result.matches))
            return facts, source_ids
        if isinstance(tool_result, ProjectSearchResult):
            facts = list(
                dict.fromkeys(
                    f"{match.project_name}: {match.summary}"
                    for match in tool_result.matches
                )
            )
            source_ids = list(dict.fromkeys(match.source_id for match in tool_result.matches))
            return facts, source_ids
        if isinstance(tool_result, ProfileQueryResult) and tool_result.source_ids:
            return list(dict.fromkeys(tool_result.value)), list(dict.fromkeys(tool_result.source_ids))
        if isinstance(tool_result, ProfileSummaryPlan):
            # Render the plan's own deterministic fact selection (D-031) instead of an
            # ad-hoc role/team_context pair — the plan already picked exactly what to say.
            catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(self._profile)}
            ordered_facts = [catalog[fact_id] for fact_id in tool_result.fact_ids if fact_id in catalog]
            facts = [fact_display_text(fact, language) for fact in ordered_facts]
            source_ids = [fact.source_id for fact in ordered_facts]
            return list(dict.fromkeys(facts)), list(dict.fromkeys(source_ids))
        if isinstance(tool_result, ResumeSearchResult):
            facts = list(dict.fromkeys(match.text for match in tool_result.matches))
            source_ids = list(dict.fromkeys(match.source_id for match in tool_result.matches))
            return facts, source_ids
        return [], []

    @staticmethod
    def _tool_has_facts(tool_result: ToolResult | None) -> bool:
        if isinstance(tool_result, ProfileQueryResult):
            return bool(tool_result.value)
        if isinstance(tool_result, ProfileSummaryPlan):
            return bool(tool_result.source_ids)
        return bool(getattr(tool_result, "matches", []))

    def _zero_selection_response(
        self,
        *,
        decision: IntentDecision,
        message: str,
        tool_name: str | None,
        tool_result_count: int,
        fallback_reason: str | None,
        state: ConversationState | None,
    ) -> AgentResponse:
        """Resolve a turn that selected no facts without ever reaching the generator.

        Recovery is topic-anchored on purpose. Where the message evidences no anchor,
        the turn clarifies instead of substituting the highest-ranked facts of an
        unrelated topic, which would answer confidently and wrongly at HTTP 200.
        """
        anchor = detect_resume_topic(self._profile, message)
        if anchor is not None:
            recovered = search_resume(
                self._profile,
                SearchResumeArguments(query=message, topic=anchor),
            )
            recovered_fact_ids = self._answer_planner.ordered_fact_ids(recovered)
            if recovered_fact_ids:
                recovery_plan = self._answer_planner.plan_from_tool(message, recovered)
                rendered = self._fact_selection_response(
                    decision=decision,
                    ordered_fact_ids=recovered_fact_ids,
                    tool_name=tool_name,
                    tool_result=recovered,
                    tool_result_count=len(recovered.matches),
                    message=message,
                    generator_skipped=True,
                    answer_plan=recovery_plan,
                    fallback_reason=fallback_reason,
                )
                if rendered is not None:
                    rendered.trace.selection_path = "recovery"
                    return rendered
        language = detect_response_language(message)
        boundary_plan = self._answer_planner.boundary_plan(message, state)
        answer = (
            "¿Podrías aclarar a qué parte del perfil profesional de Marco te refieres?"
            if language == "es"
            else "Could you clarify which part of Marco's professional profile you mean?"
        )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status="clarification",
                fallback_reason=fallback_reason,
                generator_skipped=True,
                selection_path="none",
                **plan_trace_fields(boundary_plan, "clarification"),
            ),
        )

    def _allowed_facts(self, selected_fact_ids: set[str]) -> list[ResumeFact]:
        """Resolve the turn's selected fact IDs into the canonical facts the generator may cite."""
        return [
            fact
            for fact in build_resume_fact_catalog(self._profile)
            if fact.fact_id in selected_fact_ids
        ]

    def _list_rendered_response(
        self,
        *,
        decision: IntentDecision,
        tool_name: str | None,
        tool_result: ProfileQueryResult,
        tool_result_count: int,
        message: str,
        answer_plan: AnswerPlan,
    ) -> AgentResponse:
        """Render a public profile projection deterministically; no model call is needed.

        A list of skills, languages, education, current role, or employers is exactly
        what the tool already selected — synthesis would only reorder or subset it, so
        it is rendered directly instead of round-tripping through the generator.
        """
        language = detect_response_language(message)
        heading = _LIST_QUERY_HEADINGS[tool_result.field][language]
        if tool_result.field == "skills":
            lines = [
                f"- {_SKILL_CATEGORY_LABELS[category][language]}: {', '.join(values)}"
                for category, values in self._profile.skills.model_dump().items()
                if values
            ]
        elif tool_result.field == "education":
            lines = [
                f"- {self._narrative_or_value(item.narrative, language, f'{item.degree} — {item.institution}')}"
                for item in self._profile.education
            ]
        elif tool_result.field == "current_role":
            lines = [
                f"- {self._narrative_or_value(item.narrative, language, f'{item.role} at {item.company}')}"
                for item in self._profile.experience
                if item.current
            ]
        else:  # languages, companies
            lines = [f"- {value}" for value in tool_result.value]
        answer = "\n".join([heading, *lines])
        output_result = evaluate_output(answer, self._profile)
        if not output_result.allowed:
            answer = (
                "Puedo ayudarte con el perfil profesional público de Marco, pero no "
                "puedo proporcionar esa información."
                if language == "es"
                else "I can help with Marco's public professional profile, but I can't "
                "provide that information."
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(
                    tool_name=tool_name,
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                    tool_result_count=tool_result_count,
                    grounding_status="list_rendered",
                    guardrail_output="blocked",
                    generator_skipped=True,
                    **plan_trace_fields(answer_plan, "canonical"),
                ),
            )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status="list_rendered",
                claim_source_ids=tool_result.source_ids,
                generator_skipped=True,
                **plan_trace_fields(answer_plan, "canonical"),
            ),
            state=self._state_from_result(tool_name, tool_result, tool_result.source_ids, message),
        )

    @staticmethod
    def _narrative_or_value(narrative: object, language: Literal["en", "es"], fallback: str) -> str:
        """Prefer the bilingual narrative already reviewed for a record, else its plain value."""
        if narrative is None:
            return fallback
        return narrative.en if language == "en" else narrative.es  # type: ignore[attr-defined]

    @staticmethod
    def _summary_field_override(message: str) -> str | None:
        """Reroute a misclassified summary request to the exact profile field it asked for."""
        normalized = message.casefold()
        for field, markers in _SUMMARY_FIELD_MARKERS.items():
            if any(marker in normalized for marker in markers):
                return field
        return None

    @staticmethod
    def _render_resume_result(result: ResumeSearchResult) -> str:
        headings = {
            "en": {
                "experience": "Verified experience from the profile:",
                "projects": "Verified projects from the profile:",
                "skills": "Verified technologies and skills from the profile:",
                "education": "Education specified in the profile:",
                "languages": "Languages specified in the profile:",
                "summary": "Professional summary from the profile:",
                "career_preferences": "Career preferences specified in the profile:",
            },
            "es": {
                "experience": "Experiencia verificada en el perfil:",
                "projects": "Proyectos verificados en el perfil:",
                "skills": "Tecnologías y habilidades verificadas en el perfil:",
                "education": "Educación especificada en el perfil:",
                "languages": "Idiomas especificados en el perfil:",
                "summary": "Resumen profesional basado en el perfil:",
                "career_preferences": "Preferencias profesionales especificadas en el perfil:",
            },
        }
        lines = [headings[result.language][result.topic]]
        lines.extend(f"- {fact_display_text(match, result.language)}" for match in result.matches)
        return "\n".join(lines)

    def _unclassified_response(
        self,
        message: str,
        state: ConversationState | None,
        fallback_reason: str | None,
    ) -> AgentResponse:
        """Deflect deterministically when nothing local can resolve what was asked.

        Every other boundary in this system answers at HTTP 200. A classifier that
        returns malformed output twice is a provider problem, and the user sees a
        frontend rendering nothing at all when it becomes a 503.
        """
        language = detect_response_language(message)
        boundary_plan = self._answer_planner.boundary_plan(message, state)
        answer = (
            "¿Podrías aclarar a qué parte del perfil profesional de Marco te refieres?"
            if language == "es"
            else "Could you clarify which part of Marco's professional profile you mean?"
        )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                grounding_status="clarification",
                fallback_reason=fallback_reason,
                **plan_trace_fields(boundary_plan, "clarification"),
            ),
            state=state,
        )

    def _referent_verdict(
        self,
        message: str,
        state: ConversationState | None,
    ) -> Literal["absent", "undelivered"] | None:
        """Classify an asserted antecedent against the profile and the discourse record.

        `The part where it says he worked at Google` asserts something was said. The
        agent could already answer "that entity is not in the profile", but had no way
        to answer "I never said that", so the turn fell through to a guess or a 503.

        Three verdicts, separated by the record (D-039):

        - `absent` — the antecedent names nothing in the profile. Every answer is
          assembled from profile facts alone, so it cannot have been said. Denied.
        - `undelivered` — it names real content the record does not show delivered.
          The premise is false but the content is real, so the answer is corrected
          rather than withheld.
        - `None` — the record shows it delivered, so the referent is genuine and the
          ordinary follow-up path owns the turn.

        Only phrases carrying their antecedent inline qualify. A bare referent ("that
        bit") names nothing checkable and stays on the clarification path.
        """
        antecedent = self._asserted_antecedent(message)
        if antecedent is None:
            return None
        probe = search_resume(self._profile, SearchResumeArguments(query=antecedent))
        if probe.profile_missing:
            return "absent"
        delivered = set(state.delivered_fact_ids) if state is not None else set()
        if any(fact.fact_id in delivered for fact in probe.matches):
            return None
        return "undelivered"

    @staticmethod
    def _asserted_antecedent(message: str) -> str | None:
        """Extract what a message claims was already said, if it says so explicitly."""
        normalized = " ".join(message.casefold().split())
        for phrase in _REFERENT_PHRASES:
            position = normalized.find(phrase)
            if position == -1:
                continue
            clause = normalized[position + len(phrase):].strip(" ,.;:¿?¡!")
            if clause:
                return clause
        return None

    def _negative_referent_response(
        self,
        message: str,
        state: ConversationState | None,
    ) -> AgentResponse:
        """Deny an antecedent naming nothing the profile contains."""
        language = detect_response_language(message)
        boundary_plan = self._answer_planner.boundary_plan(message, state)
        # The original casing carries the entity signal the normalized clause lost.
        entities = find_unknown_entities(self._profile, message)
        if entities:
            named = ", ".join(entities)
            answer = (
                f"No he dicho nada sobre {named}, y no aparece en el perfil de Marco, "
                "así que no hay nada a lo que pueda referirme."
                if language == "es"
                else f"I haven't said anything about {named}, and it isn't in Marco's "
                "profile, so there's nothing for me to refer back to."
            )
        else:
            answer = (
                "No he dicho eso: nada en el perfil de Marco corresponde a esa parte, "
                "así que no hay nada a lo que pueda referirme."
                if language == "es"
                else "I haven't said that — nothing in Marco's profile corresponds to "
                "it, so there's nothing for me to refer back to."
            )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                grounding_status="referent_missing",
                **plan_trace_fields(boundary_plan, "negative_referent"),
            ),
            state=state,
        )

    def _profile_missing_response(
        self,
        decision: IntentDecision,
        result: ResumeSearchResult,
        tool_name: str,
        answer_plan: AnswerPlan,
    ) -> AgentResponse:
        # `unmatched_terms` is raw query residue, not entities — naming it produced
        # "I couldn't find anything about his". Entity naming has exactly one source.
        missing_entities = find_unknown_entities(self._profile, result.query)
        if missing_entities:
            entities = ", ".join(missing_entities)
            answer = (
                f"No encontré nada sobre {entities} en el perfil de Marco, así que no "
                "puedo opinar al respecto."
                if result.language == "es"
                else f"I couldn't find anything about {entities} in Marco's profile, so "
                "I can't comment on it."
            )
        else:
            answer = (
                "No está especificado en el perfil."
                if result.language == "es"
                else "That information is not specified in the profile."
            )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                grounding_status="profile_missing",
                **plan_trace_fields(answer_plan, "canonical_not_found"),
            ),
            state=ConversationState(
                last_topic=result.topic,
                last_tool=tool_name,
                response_language=result.language,
            ),
        )

    def _follow_up_exhausted_response(
        self,
        *,
        decision: IntentDecision,
        arguments: SearchResumeArguments,
        answer_plan: AnswerPlan,
        state: ConversationState | None,
    ) -> AgentResponse:
        """Name the focused unit when all of its canonical facts were delivered."""
        catalog = build_resume_fact_catalog(self._profile)
        focused_facts = [
            fact
            for fact in catalog
            if any(
                self._sources_related(fact.source_id, source_id)
                for source_id in arguments.source_ids
            )
        ]
        entity = next((fact.entity for fact in focused_facts if fact.entity), None)
        unit_name = entity or arguments.source_ids[0]
        language = arguments.response_language or detect_response_language(arguments.query)
        answer = (
            f"Ya compartí toda la información disponible sobre {unit_name} en el perfil de Marco."
            if language == "es"
            else f"I've shared all the information available about {unit_name} in Marco's profile."
        )
        return AgentResponse(
            answer=answer,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name="search_resume",
                grounding_status="exhausted",
                answer_mode=AnswerMode.DIRECT.value,
                rendering_mode="follow_up_exhausted",
                answer_topic=answer_plan.topic,
                answer_scope=answer_plan.scope,
                requested_field=answer_plan.requested_field,
            ),
            state=ConversationState(response_language=language),
        )

    def _state_from_result(
        self,
        tool_name: str | None,
        tool_result: ToolResult | None,
        source_ids: list[str],
        message: str,
    ) -> ConversationState | None:
        topic = self._state_topic(tool_result)
        if topic is None:
            return None
        catalog = build_resume_fact_catalog(self._profile)
        verified_source_ids = list(
            dict.fromkeys(
                source_id
                for source_id in source_ids
                if any(
                    self._sources_related(fact.source_id, source_id)
                    for fact in catalog
                )
            )
        )
        if not verified_source_ids:
            return None
        entities = list(
            dict.fromkeys(
                fact.entity
                for fact in catalog
                if fact.topic == topic
                and fact.entity
                and any(
                    self._sources_related(fact.source_id, source_id)
                    for source_id in verified_source_ids
                )
            )
        )
        return ConversationState(
            last_topic=topic,
            last_source_ids=verified_source_ids[:20],
            last_entities=entities[:8],
            last_tool=tool_name,
            response_language=detect_response_language(message),
        )

    @staticmethod
    def _state_topic(tool_result: ToolResult | None) -> ResumeTopic | None:
        """Map typed tool scope to one canonical fact-catalog topic."""
        if isinstance(tool_result, ResumeSearchResult):
            return tool_result.topic
        if isinstance(tool_result, ProjectSearchResult):
            return "projects"
        if isinstance(tool_result, ExperienceFilterResult):
            return "experience"
        if isinstance(tool_result, ProfileSummaryPlan):
            return "summary"
        if isinstance(tool_result, ProfileQueryResult):
            return {
                "skills": "skills",
                "languages": "languages",
                "education": "education",
                "current_role": "experience",
                "companies": "experience",
            }.get(tool_result.field)
        return None

    @staticmethod
    def _sources_related(left: str, right: str) -> bool:
        """Relate a canonical entity source with one accepted base/highlight source."""
        return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")

    def _resume_search_arguments(
        self,
        message: str,
        state: ConversationState | None,
    ) -> SearchResumeArguments | None:
        """Return a universal plan only for unmistakably resume-related input."""
        probe = search_resume(self._profile, SearchResumeArguments(query=message, limit=1))
        normalized = " ".join(message.casefold().split())
        resume_markers = (
            "experience", "experiencia", "project", "proyecto", "built", "constru",
            "stack", "technolog", "tecnolog", "skill", "habilidad", "education",
            "educacion", "language", "idioma", "profile", "perfil", "yourself",
            "about marco", "puesto", "career preference", "sybil",
        )
        if not any(marker in normalized for marker in resume_markers):
            return None
        return SearchResumeArguments(query=message, topic=probe.topic)

    def _follow_up_plan(
        self,
        message: str,
        state: ConversationState | None,
        history: list[object] | None = None,
    ) -> SearchResumeArguments | Literal["clarify"] | None:
        normalized = normalize_resume_text(message)
        is_work_pivot = "en tu trabajo" in normalized or "at work" in normalized
        is_progressive = self._is_progressive_follow_up(message)
        is_follow_up = is_progressive or any(
            phrase in normalized
            for phrase in (
                "con que lo construiste", "y en tu trabajo", "and at work",
                "for that", "on that", "that one", "in that",
                "con eso", "en ese", "para eso", "de eso",
            )
        )
        if not is_follow_up:
            return None
        if is_work_pivot:
            return SearchResumeArguments(query=message, topic="experience")
        if is_progressive:
            explicit = self._explicit_follow_up_unit(message)
            if explicit is not None:
                entity, topic, source_id = explicit
                return SearchResumeArguments(
                    query=entity,
                    topic=topic,
                    source_ids=[source_id],
                    exclude_fact_ids=state.delivered_fact_ids if state else [],
                    response_language=detect_response_language(message),
                    limit=1,
                )
            focused = self._focused_follow_up_unit(state)
            if focused is None:
                history_plan = self._history_entity_plan(message, history or [])
                if history_plan is not None:
                    return history_plan
                return "clarify"
            entity, topic, source_id = focused
            return SearchResumeArguments(
                query=entity,
                topic=topic,
                source_ids=[source_id],
                exclude_fact_ids=state.delivered_fact_ids if state else [],
                response_language=detect_response_language(message),
                limit=1,
            )
        if (
            state is None
            or not state.last_topic
            or len(state.last_entities) != 1
            or not state.last_source_ids
        ):
            history_plan = self._history_entity_plan(message, history or [])
            if history_plan is not None:
                return history_plan
            return "clarify"
        source_roots = list(
            dict.fromkeys(source_id.split(".highlight:", 1)[0] for source_id in state.last_source_ids)
        )
        if len(source_roots) != 1:
            return "clarify"
        return SearchResumeArguments(
            query=state.last_entities[0],
            topic=state.last_topic,
            source_ids=source_roots,
        )

    @staticmethod
    def _is_progressive_follow_up(message: str) -> bool:
        normalized = normalize_resume_text(message)
        return any(
            phrase in normalized
            for phrase in (
                "tell me more", "more about", "what else", "que mas",
                "cuentame mas", "platicame mas", "dime mas",
            )
        )

    def _explicit_follow_up_unit(
        self,
        message: str,
    ) -> tuple[str, ResumeTopic, str] | None:
        """Resolve one unit named in this message, independently of conversation focus."""
        normalized = f" {normalize_resume_text(message)} "
        matches: dict[str, tuple[str, ResumeTopic, str]] = {}
        for fact in build_resume_fact_catalog(self._profile):
            if not fact.entity:
                continue
            entity = normalize_resume_text(fact.entity)
            if not entity or f" {entity} " not in normalized:
                continue
            source_id = fact.source_id.split(".highlight:", 1)[0]
            matches[source_id] = (fact.entity, fact.topic, source_id)
        return next(iter(matches.values())) if len(matches) == 1 else None

    def _focused_unit_is_exhausted(self, arguments: SearchResumeArguments) -> bool:
        """Decide exhaustion from catalog membership, never lexical search residue."""
        unit_fact_ids = {
            fact.fact_id
            for fact in build_resume_fact_catalog(self._profile)
            if fact.topic == arguments.topic
            and any(
                self._sources_related(fact.source_id, source_id)
                for source_id in arguments.source_ids
            )
        }
        return bool(unit_fact_ids) and unit_fact_ids <= set(arguments.exclude_fact_ids)

    def _focused_follow_up_unit(
        self,
        state: ConversationState | None,
    ) -> tuple[str, ResumeTopic, str] | None:
        """Resolve the accumulated focus even when the last-turn snapshot is empty."""
        if state is None or state.focus_source_id is None:
            return None
        focused = [
            fact
            for fact in build_resume_fact_catalog(self._profile)
            if self._sources_related(fact.source_id, state.focus_source_id)
        ]
        entities = list(dict.fromkeys(fact.entity for fact in focused if fact.entity))
        topics = list(dict.fromkeys(fact.topic for fact in focused))
        if len(entities) != 1 or len(topics) != 1:
            return None
        return entities[0], topics[0], state.focus_source_id

    _HISTORY_TECHNOLOGY_MARKERS = (
        "technolog", "tecnolog", "stack", "built with", "con que",
    )

    def _history_entity_plan(
        self,
        message: str,
        history: list[object],
    ) -> SearchResumeArguments | None:
        """Resolve a follow-up referent ("that") from the last few history turns.

        Only an unambiguous, single, profile-known entity mentioned recently may be
        used — zero or multiple candidates fail closed to the caller's "clarify".
        """
        if not history:
            return None
        catalog = build_resume_fact_catalog(self._profile)
        name_tokens = {
            normalize_resume_text(part) for part in self._profile.personal.name.split() if part
        }
        entity_topic: dict[str, tuple[str, ResumeTopic]] = {}
        for fact in catalog:
            if not fact.entity:
                continue
            key = normalize_resume_text(fact.entity)
            if not key or key in name_tokens:
                continue
            entity_topic.setdefault(key, (fact.entity, fact.topic))
        found: dict[str, tuple[str, ResumeTopic]] = {}
        for item in history[-4:]:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not content:
                continue
            normalized_content = normalize_resume_text(str(content))
            for key, (entity, topic) in entity_topic.items():
                if re.search(rf"\b{re.escape(key)}\b", normalized_content):
                    found[key] = (entity, topic)
        if len(found) != 1:
            return None
        entity, topic = next(iter(found.values()))
        normalized_message = normalize_resume_text(message)
        if any(marker in normalized_message for marker in self._HISTORY_TECHNOLOGY_MARKERS):
            return SearchResumeArguments(query=entity, topic=topic)
        return SearchResumeArguments(query=f"{entity} {message}", topic=None)

    def _bounded_intent_fallback(self, message: str) -> IntentDecision | None:
        """Recover only unmistakable profile intents after local classifier JSON failure."""
        normalized = " ".join(message.casefold().split())
        if self._answer_planner.synthesis_dimension(message) is not None:
            return IntentDecision(
                intent=Intent.SUMMARY_REQUEST,
                confidence=1.0,
                audience="recruiter",
            )
        if "summarize" in normalized and any(
            subject in normalized for subject in ("experience", "profile", "career")
        ):
            return IntentDecision(
                intent=Intent.SUMMARY_REQUEST,
                confidence=1.0,
                audience="recruiter",
            )
        if self._is_employment_history_question(message):
            return IntentDecision(
                intent=Intent.DIRECT_QUESTION,
                confidence=1.0,
                profile_field="companies",
            )
        if self._is_explicit_project_question(message):
            terms = self._project_fallback_terms(message)
            return IntentDecision(
                intent=Intent.SEARCH_QUERY,
                confidence=1.0,
                query=terms[0],
            )
        if "security" in normalized and any(
            subject in normalized for subject in ("work", "experience", "done")
        ):
            return IntentDecision(
                intent=Intent.FILTER_REQUEST,
                confidence=1.0,
                filter_by="tag",
                filter_value="security",
            )
        return None

    @classmethod
    def _is_explicit_project_question(cls, message: str) -> bool:
        """Require project wording plus a bounded AI/data concept before rerouting."""
        normalized = message.casefold()
        return (
            "project" in normalized
            and bool(cls._project_fallback_terms(message))
        )

    @staticmethod
    def _project_fallback_terms(message: str) -> list[str]:
        """Reduce a failed broad project query to bounded profile-domain concepts.

        Tokens are extracted with a word regex (not a plain whitespace split) so
        trailing punctuation — "AI?", "AI." — does not hide a bounded term (D-031).
        """
        normalized = message.casefold()
        tokens = re.findall(r"[a-z0-9]+", normalized)
        terms: list[str] = []
        if "ai" in tokens or "artificial intelligence" in normalized:
            terms.append("AI")
        if "rag" in tokens:
            terms.append("rag")
        if "llm" in tokens:
            terms.append("llm")
        if "retrieval" in tokens:
            terms.append("retrieval")
        if "data platform" in normalized or "data platforms" in normalized:
            terms.append("data")
        return terms

    @staticmethod
    def _is_employment_history_question(message: str) -> bool:
        """Recognize a small set of general employer-history phrasings."""
        normalized = normalize_resume_text(message)
        return (
            ("where has " in normalized and " worked" in normalized)
            or ("where did " in normalized and " work" in normalized)
            or any(
                phrase in normalized
                for phrase in (
                    "worked so far",
                    "work history",
                    "employment history",
                    "which companies",
                    "what companies",
                    "past employers",
                    "previous employers",
                    "en que empresas",
                    "para que empresas",
                    "donde ha trabajado",
                    "que empleadores",
                    "que companias",
                )
            )
        )

    @staticmethod
    def _explicit_profile_field(message: str) -> str | None:
        """Return an unmistakable current-message profile projection, if any."""
        normalized = normalize_resume_text(message)
        for field, markers in _SUMMARY_FIELD_MARKERS.items():
            if any(marker in normalized for marker in markers):
                return field
        if any(marker in normalized for marker in (
            "current role", "current job", "puesto actual", "trabajo actual",
        )):
            return "current_role"
        if AgentService._is_employment_history_question(message):
            return "companies"
        return None
