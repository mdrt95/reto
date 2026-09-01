"""Bounded orchestration that keeps model output behind typed policies."""

from typing import Literal, Protocol

from src.agent.contracts import (
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
)
from src.agent.grounding import profile_source_ids, verify_claims
from src.guardrails.input_guard import evaluate_input
from src.guardrails.output_guard import evaluate_output
from src.models.profile import Profile
from src.tools.profile_tools import (
    ExperienceFilterResult,
    ProfileQueryResult,
    ProfileSummaryPlan,
    ProjectSearchResult,
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
    find_unknown_entities,
    query_profile,
    search_projects,
    search_resume,
    summarize_profile,
)

_RANKING_MARKERS = (
    "rank", "ranking", "best to worst", "worst to best",
    "ordena", "clasifica", "mejor a peor", "peor a mejor", "del mejor",
)


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
        profile: Profile,
        tool_result: object | None,
        allowed_source_ids: set[str],
        allowed_fact_ids: set[str] | None = None,
    ) -> GeneratedResponse: ...


ToolResult = ExperienceFilterResult | ProfileQueryResult | ProfileSummaryPlan | ProjectSearchResult | ResumeSearchResult


class AgentService:
    """Run one bounded, traceable profile-answering workflow per chat turn."""

    def __init__(self, *, profile: Profile, classifier: IntentClassifier, generator: ResponseGenerator) -> None:
        self._profile = profile
        self._classifier = classifier
        self._generator = generator

    def respond(
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

        unknown_entities = find_unknown_entities(self._profile, message)
        if unknown_entities:
            language = detect_response_language(message)
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
                trace=AgentTrace(grounding_status="profile_missing"),
            )

        follow_up = self._follow_up_plan(message, state)
        if follow_up == "clarify":
            language = detect_response_language(message)
            answer = (
                "¿A qué parte o elemento anterior te refieres?"
                if language == "es"
                else "Which part or item from the previous answer do you mean?"
            )
            return AgentResponse(
                answer=answer,
                trace=AgentTrace(grounding_status="clarification"),
                state=state,
            )

        normalized_message = " ".join(message.casefold().split())
        if any(marker in normalized_message for marker in _RANKING_MARKERS):
            language = detect_response_language(message)
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
                trace=AgentTrace(grounding_status="clarification"),
            )

        try:
            decision = self._classifier.classify(message, history)
        except GenerationUnavailableError:
            decision = self._bounded_intent_fallback(message)
            if decision is None:
                if follow_up is None and self._resume_search_arguments(message, state) is None:
                    raise
                decision = IntentDecision(intent=Intent.FOLLOW_UP, confidence=1.0)
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
                ),
            )
        if decision.confidence < 0.7 and self._resume_search_arguments(message, state) is None:
            language = detect_response_language(message)
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
        if isinstance(tool_result, ProfileQueryResult):
            tool_result_count = len(tool_result.value)
        else:
            tool_result_count = len(getattr(tool_result, "matches", [])) if tool_result else 0
        if isinstance(tool_result, ResumeSearchResult) and tool_result.profile_missing:
            return self._profile_missing_response(decision, tool_result, tool_name)
        allowed_sources = profile_source_ids(self._profile)
        selected_fact_ids = self._selected_fact_ids(tool_result)
        try:
            generated = self._generator.generate(
                message=message,
                history=history,
                profile=self._profile,
                tool_result=tool_result,
                allowed_source_ids=allowed_sources,
                allowed_fact_ids=selected_fact_ids,
            )
        except GenerationUnavailableError:
            fallback = self._tool_fallback_response(
                decision=decision,
                tool_name=tool_name,
                tool_result=tool_result,
                tool_result_count=tool_result_count,
                message=message,
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
            rendered = self._fact_selection_response(
                decision=decision,
                grounding=grounding,
                tool_name=tool_name,
                tool_result=tool_result,
                tool_result_count=tool_result_count,
                message=message,
            )
            if rendered is not None:
                return rendered
        if grounding.status != "fully_grounded":
            try:
                generated = self._generator.generate(
                    message=message,
                    history=history,
                    profile=self._profile,
                    tool_result=tool_result,
                    allowed_source_ids=allowed_sources,
                    allowed_fact_ids=selected_fact_ids,
                )
            except GenerationUnavailableError:
                fallback = self._tool_fallback_response(
                    decision=decision,
                    tool_name=tool_name,
                    tool_result=tool_result,
                    tool_result_count=tool_result_count,
                    message=message,
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
            if not verified_facts:
                verified_facts, accepted_source_ids = self._verified_tool_facts(tool_result)
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
                    ),
                )
            answer = "\n\n".join(verified_facts)

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
            ),
            state=self._state_from_result(tool_name, tool_result, accepted_source_ids, message),
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
            filter_by = decision.filter_by if decision.filter_by in {"technology", "tag", "role"} else "tag"
            result = filter_experience(
                self._profile,
                FilterExperienceArguments(filter_by=filter_by, value=decision.filter_value or "profile"),
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
            audience = decision.audience if decision.audience in {"technical", "recruiter", "executive"} else "recruiter"
            return "summarize_profile", summarize_profile(
                self._profile,
                SummarizeProfileArguments(audience=audience),
            )
        profile_field = decision.profile_field
        if (
            decision.intent in {Intent.DIRECT_QUESTION, Intent.FOLLOW_UP}
            and profile_field is None
            and self._is_employment_history_question(message)
        ):
            profile_field = "companies"
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

    def _tool_fallback_response(
        self,
        *,
        decision: IntentDecision,
        tool_name: str | None,
        tool_result: ToolResult | None,
        tool_result_count: int,
        message: str,
    ) -> AgentResponse | None:
        """Return verified deterministic facts after model generation is unavailable."""
        verified_facts, accepted_source_ids = self._verified_tool_facts(tool_result)
        if not verified_facts:
            return None
        answer = (
            self._render_resume_result(tool_result)
            if isinstance(tool_result, ResumeSearchResult)
            else "\n\n".join(verified_facts)
        )
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
            ),
            state=self._state_from_result(tool_name, tool_result, accepted_source_ids, message),
        )

    def _fact_selection_response(
        self,
        *,
        decision: IntentDecision,
        grounding: GroundingResult,
        tool_name: str | None,
        tool_result: ToolResult | None,
        tool_result_count: int,
        message: str,
    ) -> AgentResponse | None:
        """Render provider-selected fact IDs exclusively from canonical fact values."""
        claim_fact_ids = grounding.claim_fact_ids
        ordered_fact_ids = list(
            dict.fromkeys(
                fact_id
                for claim_index in sorted(claim_fact_ids)
                for fact_id in claim_fact_ids[claim_index]
            )
        )
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
                grounding_status="fact_rendered",
                claim_source_ids=source_ids,
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

    def _verified_tool_facts(self, tool_result: ToolResult | None) -> tuple[list[str], list[str]]:
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
            selected_sources = set(tool_result.source_ids)
            facts = [
                fact
                for experience in self._profile.experience
                if f"experience:{experience.id}" in selected_sources
                for fact in (
                    f"{experience.role} at {experience.company}",
                    experience.team_context,
                )
            ]
            source_ids = [
                f"experience:{experience.id}"
                for experience in self._profile.experience
                if f"experience:{experience.id}" in selected_sources
            ]
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

    def _selected_fact_ids(self, tool_result: ToolResult | None) -> set[str]:
        if isinstance(tool_result, ResumeSearchResult):
            return {match.fact_id for match in tool_result.matches}
        _, source_ids = self._verified_tool_facts(tool_result)
        return {
            fact.fact_id
            for fact in build_resume_fact_catalog(self._profile)
            if any(
                fact.source_id == source_id
                or fact.source_id.startswith(f"{source_id}.")
                or source_id.startswith(f"{fact.source_id}.")
                for source_id in source_ids
            )
        }

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
        lines.extend(f"- {match.text}" for match in result.matches)
        return "\n".join(lines)

    def _profile_missing_response(
        self,
        decision: IntentDecision,
        result: ResumeSearchResult,
        tool_name: str,
    ) -> AgentResponse:
        if result.unmatched_terms:
            entities = ", ".join(result.unmatched_terms)
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
            ),
            state=ConversationState(
                last_topic=result.topic,
                last_tool=tool_name,
                response_language=result.language,
            ),
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
    ) -> SearchResumeArguments | Literal["clarify"] | None:
        normalized = " ".join(message.casefold().split())
        is_work_pivot = "en tu trabajo" in normalized or "at work" in normalized
        is_follow_up = any(
            phrase in normalized
            for phrase in (
                "con qué lo construiste", "con que lo construiste", "tell me more about that one",
                "what else", "qué más", "que mas", "y en tu trabajo", "and at work",
            )
        )
        if not is_follow_up:
            return None
        if is_work_pivot:
            return SearchResumeArguments(query=message, topic="experience")
        if state is None or not state.last_topic:
            return "clarify"
        if len(state.last_entities) != 1 or not state.last_source_ids:
            return "clarify"
        source_roots = list(
            dict.fromkeys(source_id.split(".highlight:", 1)[0] for source_id in state.last_source_ids)
        )
        if len(source_roots) != 1:
            return "clarify"
        if "what else" in normalized or "qué más" in normalized or "que mas" in normalized:
            return SearchResumeArguments(
                query=state.last_entities[0],
                topic=state.last_topic,
                source_ids=source_roots,
                exclude_source_ids=state.last_source_ids,
            )
        if "tell me more" in normalized:
            return SearchResumeArguments(
                query=state.last_entities[0],
                topic=state.last_topic,
                source_ids=source_roots,
                exclude_source_ids=state.last_source_ids,
            )
        return SearchResumeArguments(
            query=state.last_entities[0],
            topic=state.last_topic,
            source_ids=source_roots,
        )

    def _bounded_intent_fallback(self, message: str) -> IntentDecision | None:
        """Recover only unmistakable profile intents after local classifier JSON failure."""
        normalized = " ".join(message.casefold().split())
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
        """Reduce a failed broad project query to bounded profile-domain concepts."""
        normalized = message.casefold()
        terms: list[str] = []
        if "ai" in normalized.split() or "artificial intelligence" in normalized:
            terms.append("AI")
        if "data platform" in normalized:
            terms.append("data")
        return terms

    @staticmethod
    def _is_employment_history_question(message: str) -> bool:
        """Recognize a small set of general employer-history phrasings."""
        normalized = " ".join(message.casefold().split())
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
                )
            )
        )
