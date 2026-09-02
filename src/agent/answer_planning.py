"""Deterministic answer planning and canonical direct-answer rendering."""

import re
from collections import Counter
from typing import Literal

from src.agent.contracts import (
    AnswerMode,
    AnswerPlan,
    ConversationState,
    MAX_SYNTHESIS_FACTS_BY_LANGUAGE,
    MAX_SYNTHESIS_SENTENCES,
    MAX_SYNTHESIS_WORDS,
    SynthesisDimension,
)
from src.models.profile import Profile
from src.tools.profile_tools import (
    ExperienceFilterResult,
    ProfileQueryResult,
    ProfileSummaryPlan,
    ProjectSearchResult,
    ResumeFact,
    ResumeSearchResult,
    ResumeTopic,
    SearchResumeArguments,
    build_resume_fact_catalog,
    detect_response_language,
    fact_display_text,
    normalize_resume_text,
    search_resume,
)

ToolResult = (
    ExperienceFilterResult
    | ProfileQueryResult
    | ProfileSummaryPlan
    | ProjectSearchResult
    | ResumeSearchResult
)

_MONTH_NAMES = {
    "en": (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ),
    "es": (
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ),
}

_SYNTHESIS_MARKERS: dict[SynthesisDimension, tuple[str, ...]] = {
    "summary": ("summarize", "summary", "resume la", "resumen"),
    # Achievement wording points at the outcome-bearing facts the impact dimension
    # already selects and already ranks; matching is substring, so plurals are covered.
    "impact": (
        "impact", "impacto", "outcome", "resultado", "efecto",
        "achiev", "accomplish", "logro",
    ),
    "significance": (
        "significance", "significancia", "why does", "why is", "por que importa",
        "importancia",
    ),
    "comparison": (
        "compare", "comparison", "versus", "difference", "compara", "comparacion",
        "diferencia",
    ),
    "explanation": ("explain", "explanation", "explica", "por que", "how did", "como"),
    "conclusion": (
        "conclusion", "conclude", "own words", "tus propias palabras",
        "show the matching work", "show matching work", "what is your experience",
        "cual es tu experiencia", "more about", "cuentame", "platicame", "hablame",
    ),
}

_SPANISH_SUMMARY_VERBS = {"resume", "resumeme", "resumen", "resumir"}

_EXPLICIT_OUTCOME_MARKERS = (
    "ahead", "beating", "resolved", "reduced", "improve", "improved", "supporting",
    "independent deployment", "availability", "deadline", "expectations",
    "antes de", "superando", "resolviendo", "reduciendo", "mejorar", "permitiendo",
)


class AnswerPlanner:
    """Select the smallest canonical fact set and its typed answer contract."""

    def __init__(self, profile: Profile) -> None:
        self._profile = profile

    def explicit_direct_plan(
        self,
        message: str,
        state: ConversationState | None = None,
    ) -> AnswerPlan | None:
        """Resolve bounded current-message evidence before advisory classifier fields.

        The order is deliberate: field/date questions are narrower than tags,
        technologies, domains, and entities. Each match therefore produces the
        smallest sufficient canonical fact set without parent/sibling expansion.
        """
        normalized = normalize_resume_text(message)
        language = detect_response_language(message)
        catalog = build_resume_fact_catalog(self._profile)
        facts_by_id = {fact.fact_id: fact for fact in catalog}
        if self.synthesis_dimension(message) is not None:
            return None

        date_field: Literal["start_date", "end_date", "current"] | None = None
        if any(marker in normalized for marker in (
            "hasta cuando", "leave", "left", "end", "ended", "termino", "finalizo",
        )):
            date_field = "end_date"
        elif any(marker in normalized for marker in (
            "desde cuando", "desde", "when did", "when has", "start", "started", "joined",
            "how long", "comenzo", "empezo", "inicio",
        )):
            date_field = "start_date"
        elif any(marker in normalized for marker in (
            "currently", "current", "still", "actualmente", "sigue", "continua",
        )):
            date_field = "current"
        if date_field is not None:
            for experience in self._profile.experience:
                if not self._message_mentions_company(normalized, experience.company):
                    continue
                return self._date_plan(experience, date_field, language, facts_by_id)
            # Only once the current message names no employer may verified state
            # supply the missing referent. An explicit entity always outranks state.
            referred = self._state_referent_company(state)
            if referred is not None:
                for experience in self._profile.experience:
                    if experience.company != referred:
                        continue
                    return self._date_plan(experience, date_field, language, facts_by_id)

        matched_tag = self.profile_tag_match(message)
        if matched_tag is not None:
            tag_facts = [
                fact
                for fact in catalog
                if fact.topic == "experience"
                and fact.source_id.startswith("experience:")
                and ".highlight:" in fact.source_id
                and normalize_resume_text(matched_tag) in {
                    normalize_resume_text(keyword) for keyword in fact.keywords
                }
            ]
            if tag_facts:
                return self._plan(
                    mode=AnswerMode.DIRECT,
                    topic="experience",
                    scope="employment",
                    requested_field="tag",
                    language=language,
                    facts=tag_facts,
                )

        technology = self.named_technology(message)
        if technology is not None:
            direct_mentions = [
                fact
                for fact in catalog
                if fact.topic in {"projects", "experience"}
                and fact.field_name is None
                and self._normalized_phrase_present(
                    normalize_resume_text(
                        " ".join(
                            filter(None, [fact.text, fact.narrative_en, fact.narrative_es])
                        )
                    ),
                    normalize_resume_text(technology),
                )
            ]
            if direct_mentions:
                topic = direct_mentions[0].topic
                direct_mentions = [fact for fact in direct_mentions if fact.topic == topic]
                return self._plan(
                    mode=AnswerMode.DIRECT,
                    topic=topic,
                    scope="project" if topic == "projects" else "employment",
                    requested_field="technology",
                    language=language,
                    facts=direct_mentions,
                )

        normalized_tokens = set(normalized.split())
        if "project" in normalized_tokens and normalized_tokens & {
            "what", "which", "que", "cuales",
        }:
            project_facts = [
                fact
                for fact in catalog
                if fact.topic == "projects"
                and ".highlight:" not in fact.source_id
            ]
            if project_facts:
                return self._plan(
                    mode=AnswerMode.DIRECT,
                    topic="projects",
                    scope="project",
                    requested_field="projects",
                    language=language,
                    facts=project_facts,
                )

        if any(phrase in normalized for phrase in (
            "dime acerca de la experiencia", "tell me about marco s experience",
            "work history", "historial laboral",
        )):
            experience_facts = [
                fact
                for fact in catalog
                if fact.topic == "experience"
                and fact.fact_id.startswith("fact:experience:")
                and ".highlight:" not in fact.source_id
                and fact.field_name is None
            ]
            named = [
                fact for fact in experience_facts
                if fact.entity and self._message_mentions_company(normalized, fact.entity)
            ]
            selected = named or experience_facts
            if selected:
                return self._plan(
                    mode=AnswerMode.DIRECT,
                    topic="experience",
                    scope="employment",
                    requested_field="experience",
                    language=language,
                    facts=selected,
                )

        for project in self._profile.projects:
            if normalize_resume_text(project.name) not in normalized:
                continue
            fact = facts_by_id.get(f"fact:project:{project.id}")
            if fact is not None:
                return self._plan(
                    mode=AnswerMode.DIRECT,
                    topic="projects",
                    scope="project",
                    requested_field="projects",
                    language=language,
                    facts=[fact],
                )
        return None

    def plan_from_tool(
        self,
        message: str,
        tool_result: ToolResult | None,
    ) -> AnswerPlan:
        """Create the typed plan for every classified in-scope tool outcome."""
        language = detect_response_language(message)
        synthesis_dimension = self.synthesis_dimension(message)
        catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(self._profile)}
        ordered_fact_ids = self.ordered_fact_ids(tool_result)
        facts = [catalog[fact_id] for fact_id in ordered_fact_ids if fact_id in catalog]
        if isinstance(tool_result, ProfileQueryResult):
            topic = {
                "skills": "skills",
                "languages": "languages",
                "education": "education",
                "current_role": "experience",
                "companies": "experience",
            }.get(tool_result.field, "summary")
            scope = {
                "skills": "skill",
                "languages": "language",
                "education": "education",
            }.get(tool_result.field, "employment")
            requested_field = "employer" if tool_result.field == "companies" else tool_result.field
            mode = AnswerMode.DIRECT
            facts = [
                fact
                for fact in catalog.values()
                if fact.topic == topic
                and fact.source_id in set(tool_result.source_ids)
                and fact.field_name is None
                and (
                    topic not in {"experience", "projects"}
                    or ".highlight:" not in fact.source_id
                )
            ]
        elif isinstance(tool_result, ProjectSearchResult):
            topic, scope = "projects", "project"
            requested_field = "technology" if self.named_technology(message) else "projects"
            mode = AnswerMode.SYNTHESIS
        elif isinstance(tool_result, ExperienceFilterResult):
            topic, scope, requested_field = "experience", "employment", "tag"
            mode = AnswerMode.SYNTHESIS
        elif isinstance(tool_result, ResumeSearchResult):
            topic = tool_result.topic
            scope = {
                "projects": "project",
                "experience": "employment",
                "skills": "skill",
                "education": "education",
                "languages": "language",
                "career_preferences": "career_preferences",
            }.get(topic, "profile")
            requested_field = {
                "projects": "projects",
                "experience": "experience",
                "skills": "skills",
                "education": "education",
                "languages": "languages",
                "career_preferences": "career_preferences",
            }.get(topic, "summary")
            mode = AnswerMode.SYNTHESIS
        else:
            topic, scope, requested_field = "summary", "profile", "summary"
            mode = AnswerMode.SYNTHESIS
        if synthesis_dimension is not None:
            topic, scope, requested_field, facts = self._synthesis_selection(
                message=message,
                dimension=synthesis_dimension,
                fallback_topic=topic,
                fallback_scope=scope,
                fallback_field=requested_field,
                tool_facts=facts,
                catalog=list(catalog.values()),
            )
            mode = AnswerMode.SYNTHESIS
        elif mode is AnswerMode.SYNTHESIS:
            # Tool type does not imply transformation. Without explicit summary,
            # impact, significance, comparison, explanation, or conclusion wording,
            # the selected canonical facts remain a direct answer.
            mode = AnswerMode.DIRECT
        return self._plan(
            mode=mode,
            topic=topic,
            scope=scope,
            requested_field=requested_field,
            language=language,
            facts=facts,
            synthesis_dimension=synthesis_dimension,
        )

    def synthesis_dimension(self, message: str) -> SynthesisDimension | None:
        """Classify only explicit transformation language into a bounded dimension."""
        normalized = normalize_resume_text(message)
        # "Resume" is a Spanish imperative verb whose object varies ("resume la
        # experiencia", "resume los proyectos"), and an English noun for the document
        # itself. Matching the bare token only in Spanish covers every object without
        # ever firing on "Marco's resume".
        if (
            detect_response_language(message) == "es"
            and set(normalized.split()) & _SPANISH_SUMMARY_VERBS
        ):
            return "summary"
        for dimension, markers in _SYNTHESIS_MARKERS.items():
            if any(marker in normalized for marker in markers):
                return dimension
        return None

    def _synthesis_selection(
        self,
        *,
        message: str,
        dimension: SynthesisDimension,
        fallback_topic: ResumeTopic,
        fallback_scope: Literal[
            "profile", "employment", "project", "skill", "education", "language",
            "career_preferences",
        ],
        fallback_field: Literal[
            "start_date", "end_date", "current", "projects", "experience", "technology",
            "tag", "employer", "skills", "education", "languages", "current_role",
            "summary", "career_preferences",
        ],
        tool_facts: list[ResumeFact],
        catalog: list[ResumeFact],
    ) -> tuple[
        ResumeTopic,
        Literal[
            "profile", "employment", "project", "skill", "education", "language",
            "career_preferences",
        ],
        Literal[
            "start_date", "end_date", "current", "projects", "experience", "technology",
            "tag", "employer", "skills", "education", "languages", "current_role",
            "summary", "career_preferences",
        ],
        list[ResumeFact],
    ]:
        """Rank a small, non-overlapping fact set for the requested dimension."""
        normalized_tokens = self._planning_tokens(message)
        if "project" in normalized_tokens:
            topic: ResumeTopic = "projects"
            scope = "project"
            requested_field = "projects"
        elif normalized_tokens & {
            "experience", "experiencia", "work", "trabajo", "career", "carrera",
        }:
            topic = "experience"
            scope = "employment"
            requested_field = "experience"
        else:
            topic = fallback_topic
            scope = fallback_scope
            requested_field = fallback_field
            if dimension == "impact":
                # An unscoped impact question ("achievements", "logros") carries no
                # topic token, so a classifier-selected tool must not choose its scope.
                # Anchor on the topic that actually holds the canonical outcome facts.
                # No fact is derived: the predicate is the same one the impact branch
                # already filters on.
                outcome_topics = Counter(
                    fact.topic
                    for fact in catalog
                    if fact.field_name is None and self._has_explicit_outcome(fact)
                )
                if outcome_topics:
                    topic = outcome_topics.most_common(1)[0][0]
                    scope = "project" if topic == "projects" else "employment"
                    requested_field = "projects" if topic == "projects" else "experience"

        topic_facts = [
            fact
            for fact in catalog
            if fact.topic == topic and fact.field_name is None
        ]
        named_technology = self.named_technology(message)
        tag = self.profile_tag_match(message)
        explicitly_scoped = False
        if named_technology is not None:
            normalized_technology = normalize_resume_text(named_technology)
            technology_facts = [
                fact
                for fact in catalog
                if fact.field_name is None
                and self._normalized_phrase_present(
                    normalize_resume_text(
                        " ".join(
                            filter(None, [fact.text, fact.narrative_en, fact.narrative_es])
                        )
                    ),
                    normalized_technology,
                )
            ]
            if technology_facts:
                # A bare skill label proves familiarity, but it cannot answer what
                # work was done or what impact it had. Prefer the canonical work
                # narrative that directly mentions the requested technology.
                work_facts = [
                    fact
                    for fact in technology_facts
                    if fact.topic in {"projects", "experience"}
                ]
                technology_facts = work_facts or technology_facts
                topic = technology_facts[0].topic
                scope = "project" if topic == "projects" else "employment"
                requested_field = "technology"
                candidates = [fact for fact in technology_facts if fact.topic == topic]
                explicitly_scoped = True
            else:
                candidates = tool_facts or topic_facts
        elif tag is not None and tool_facts:
            candidates = tool_facts
            explicitly_scoped = True
        else:
            candidates = topic_facts or tool_facts
        if dimension == "impact":
            original_candidates = candidates
            outcome_candidates = [
                fact for fact in original_candidates if self._has_explicit_outcome(fact)
            ]
            candidates = outcome_candidates or (original_candidates if explicitly_scoped else [])
        elif (
            dimension in {"significance", "comparison", "explanation", "conclusion"}
            and tool_facts
            and all(fact.topic == topic for fact in tool_facts)
        ):
            # An explicit tool filter/query is a narrower evidence boundary than the
            # surrounding topic. Transform that selection; do not reopen the profile.
            candidates = tool_facts
        elif topic == "experience" and dimension == "summary":
            # Coordination is valid evidence but less representative than concrete delivery,
            # security, performance, and architecture work for a bounded overview.
            substantive = [
                fact
                for fact in candidates
                if not fact.text.casefold().startswith("assisted ")
            ]
            candidates = substantive or candidates

        candidates = sorted(
            candidates,
            key=lambda fact: (
                -self._synthesis_rank(fact, dimension),
                fact.fact_id,
            ),
        )

        detail_requested = self.detail_requested(message)
        # Beyond this bound the 3-sentence/75-word delivery budget is unreachable, so
        # the extra evidence could only ever be discarded by a length rejection.
        language_limit = MAX_SYNTHESIS_FACTS_BY_LANGUAGE[detect_response_language(message)]
        selection_limit = (
            min(2, language_limit)
            if dimension in {"significance", "conclusion"} and not detail_requested
            else language_limit
        )
        bounded: list[ResumeFact] = []
        for fact in candidates:
            if any(self._semantic_overlap(fact, selected) for selected in bounded):
                continue
            bounded.append(fact)
            if len(bounded) == selection_limit:
                break
        return topic, scope, requested_field, bounded

    @classmethod
    def detail_requested(cls, message: str) -> bool:
        """Return whether the current request explicitly expands synthesis detail."""
        return bool(
            cls._planning_tokens(message)
            & {"detail", "details", "detailed", "detalle", "detalles", "detallado"}
        )

    @staticmethod
    def _planning_tokens(message: str) -> set[str]:
        aliases = {
            "projects": "project",
            "proyectos": "project",
            "proyecto": "project",
            "seguridad": "security",
        }
        return {
            aliases.get(token.rstrip("."), token.rstrip("."))
            for token in normalize_resume_text(message).split()
        }

    @staticmethod
    def _has_explicit_outcome(fact: ResumeFact) -> bool:
        text = normalize_resume_text(
            " ".join(filter(None, [fact.text, fact.narrative_en, fact.narrative_es]))
        )
        return any(marker in text for marker in _EXPLICIT_OUTCOME_MARKERS)

    @classmethod
    def has_explicit_outcome(cls, fact: ResumeFact) -> bool:
        """Expose the canonical impact predicate to the orchestration boundary."""
        return cls._has_explicit_outcome(fact)

    @classmethod
    def _synthesis_rank(
        cls,
        fact: ResumeFact,
        dimension: SynthesisDimension,
    ) -> int:
        """Score evidence by requested meaning, independent of profile list order."""
        normalized = normalize_resume_text(
            " ".join(filter(None, [fact.text, fact.narrative_en, fact.narrative_es]))
        )
        score = 0
        if ".highlight:" not in fact.source_id:
            score += 100
        if cls._has_explicit_outcome(fact):
            score += 60 if dimension in {"impact", "significance"} else 25
        if any(character.isdigit() for character in normalized):
            score += 20
        score += min(20, len(set(fact.keywords)) * 2)
        return score

    @staticmethod
    def _semantic_overlap(left: ResumeFact, right: ResumeFact) -> bool:
        if left.source_id == right.source_id:
            return True
        left_tokens = set(normalize_resume_text(left.text).split())
        right_tokens = set(normalize_resume_text(right.text).split())
        if not left_tokens or not right_tokens:
            return False
        similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        return similarity >= 0.72

    def ordered_fact_ids(self, tool_result: ToolResult | None) -> list[str]:
        """Return the tool-selected canonical facts in deterministic source order."""
        if isinstance(tool_result, ResumeSearchResult):
            return [match.fact_id for match in tool_result.matches]
        if isinstance(tool_result, ProfileSummaryPlan):
            return list(dict.fromkeys(tool_result.fact_ids))
        if isinstance(tool_result, (ExperienceFilterResult, ProjectSearchResult)):
            source_ids = [match.source_id for match in tool_result.matches]
        elif isinstance(tool_result, ProfileQueryResult):
            source_ids = tool_result.source_ids
        else:
            source_ids = []
        if not source_ids:
            return []
        catalog = build_resume_fact_catalog(self._profile)
        ordered: list[str] = []
        for source_id in dict.fromkeys(source_ids):
            for fact in catalog:
                if fact.fact_id in ordered:
                    continue
                if fact.source_id == source_id and fact.field_name is None:
                    ordered.append(fact.fact_id)
        return ordered

    def _date_plan(
        self,
        experience: object,
        date_field: Literal["start_date", "end_date", "current"],
        language: Literal["en", "es"],
        facts_by_id: dict[str, ResumeFact],
    ) -> AnswerPlan:
        """Build the smallest canonical fact set answering one employment date."""
        fact_ids = [f"fact:experience:{experience.id}:{date_field}"]
        if date_field == "start_date":
            fact_ids.append(f"fact:experience:{experience.id}:current")
        return self._plan(
            mode=AnswerMode.DIRECT,
            topic="experience",
            scope="employment",
            requested_field=date_field,
            language=language,
            facts=[facts_by_id[fact_id] for fact_id in fact_ids],
        )

    def _state_referent_company(self, state: ConversationState | None) -> str | None:
        """Resolve exactly one profile employer from verified state, or nothing.

        Zero entities, several entities, and an entity that is not a single profile
        employer all fail closed. An ambiguous referent must reach clarification;
        guessing one of several employers is how a confident wrong answer happens.
        """
        if state is None or len(state.last_entities) != 1:
            return None
        entity = normalize_resume_text(state.last_entities[0])
        companies = [
            experience.company
            for experience in self._profile.experience
            if normalize_resume_text(experience.company) == entity
        ]
        return companies[0] if len(companies) == 1 else None

    def boundary_plan(
        self,
        message: str,
        state: ConversationState | None,
    ) -> AnswerPlan:
        """Type safe deterministic clarification/not-found responses with no facts."""
        probe = search_resume(
            self._profile,
            SearchResumeArguments(query=message, limit=1),
        )
        topic = state.last_topic if state and state.last_topic else probe.topic
        scope = {
            "projects": "project",
            "experience": "employment",
            "skills": "skill",
            "education": "education",
            "languages": "language",
            "career_preferences": "career_preferences",
        }.get(topic, "profile")
        requested_field = {
            "projects": "projects",
            "experience": "experience",
            "skills": "skills",
            "education": "education",
            "languages": "languages",
            "career_preferences": "career_preferences",
        }.get(topic, "summary")
        return self._plan(
            mode=AnswerMode.DIRECT,
            topic=topic,
            scope=scope,
            requested_field=requested_field,
            language=detect_response_language(message),
            facts=[],
        )

    def named_technology(self, message: str) -> str | None:
        """Return the longest profile-defined technology named in the message."""
        normalized_message = normalize_resume_text(message)
        candidates: list[str] = []
        for project in self._profile.projects:
            candidates.extend(project.technologies)
            for highlight in project.highlights:
                candidates.extend(highlight.technologies)
        for experience in self._profile.experience:
            for highlight in experience.highlights:
                candidates.extend(highlight.technologies)
        matches = [
            value
            for value in dict.fromkeys(candidates)
            if normalize_resume_text(value)
            and self._normalized_phrase_present(
                normalized_message, normalize_resume_text(value)
            )
        ]
        return max(matches, key=lambda value: len(normalize_resume_text(value)), default=None)

    def profile_tag_match(self, message: str) -> str | None:
        """Return the first profile-defined experience-highlight tag in the message."""
        normalized_message = normalize_resume_text(message)
        message_tokens = set(normalized_message.split())
        for experience in self._profile.experience:
            for highlight in experience.highlights:
                for tag in highlight.tags:
                    normalized_tag = normalize_resume_text(tag)
                    if not normalized_tag:
                        continue
                    if " " in normalized_tag:
                        if normalized_tag in normalized_message:
                            return tag
                    elif normalized_tag in message_tokens:
                        return tag
        return None

    @staticmethod
    def _plan(
        *,
        mode: AnswerMode,
        topic: ResumeTopic,
        scope: Literal[
            "profile", "employment", "project", "skill", "education", "language",
            "career_preferences",
        ],
        requested_field: Literal[
            "start_date", "end_date", "current", "projects", "experience", "technology",
            "tag", "employer", "skills", "education", "languages", "current_role",
            "summary", "career_preferences",
        ],
        language: Literal["en", "es"],
        facts: list[ResumeFact],
        synthesis_dimension: SynthesisDimension | None = None,
    ) -> AnswerPlan:
        unique_facts = list({fact.fact_id: fact for fact in facts}.values())
        return AnswerPlan(
            mode=mode,
            topic=topic,
            scope=scope,
            requested_field=requested_field,
            language=language,
            synthesis_dimension=synthesis_dimension,
            selected_fact_ids=[fact.fact_id for fact in unique_facts],
            selected_source_ids=list(dict.fromkeys(fact.source_id for fact in unique_facts)),
        )

    @staticmethod
    def _message_mentions_company(normalized_message: str, company: str) -> bool:
        normalized_company = normalize_resume_text(company)
        aliases = [normalized_company]
        aliases.extend(
            normalize_resume_text(part)
            for part in re.split(r"[()]", company)
            if len(normalize_resume_text(part).split()) >= 2
        )
        return any(alias and alias in normalized_message for alias in aliases)

    @staticmethod
    def _normalized_phrase_present(haystack: str, needle: str) -> bool:
        haystack_tokens = haystack.split()
        needle_tokens = needle.split()
        width = len(needle_tokens)
        return bool(width) and any(
            haystack_tokens[index:index + width] == needle_tokens
            for index in range(len(haystack_tokens) - width + 1)
        )


class DirectAnswerRenderer:
    """Render an already-authorized direct plan from canonical profile facts."""

    def __init__(self, profile: Profile) -> None:
        self._profile = profile

    def render(self, plan: AnswerPlan) -> str:
        if plan.requested_field in {"start_date", "end_date", "current"}:
            return self._render_employment_field(plan)
        catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(self._profile)}
        return "\n\n".join(
            fact_display_text(catalog[fact_id], plan.language)
            for fact_id in plan.selected_fact_ids
        )

    def _render_employment_field(self, plan: AnswerPlan) -> str:
        source_id = plan.selected_source_ids[0]
        experience_id = source_id.removeprefix("experience:")
        experience = next(item for item in self._profile.experience if item.id == experience_id)
        start = self._format_partial_date(experience.start_date, plan.language)
        end = (
            self._format_partial_date(experience.end_date, plan.language)
            if experience.end_date
            else None
        )
        if plan.language == "es":
            if plan.requested_field == "start_date":
                status = " y actualmente continúa en ese puesto" if experience.current else ""
                return f"Marco trabaja en {experience.company} desde {start}{status}."
            if plan.requested_field == "end_date":
                return (
                    f"Marco trabajó en {experience.company} hasta {end}."
                    if end else f"Marco trabaja actualmente en {experience.company}; no hay fecha de fin."
                )
            return (
                f"Sí, Marco trabaja actualmente en {experience.company}."
                if experience.current else f"No, Marco ya no trabaja en {experience.company}."
            )
        if plan.requested_field == "start_date":
            status = " and currently remains in that role" if experience.current else ""
            return f"Marco has worked at {experience.company} since {start}{status}."
        if plan.requested_field == "end_date":
            return (
                f"Marco worked at {experience.company} until {end}."
                if end else f"Marco currently works at {experience.company}; no end date is listed."
            )
        return (
            f"Yes, Marco currently works at {experience.company}."
            if experience.current else f"No, Marco no longer works at {experience.company}."
        )

    @staticmethod
    def _format_partial_date(value: str | None, language: Literal["en", "es"]) -> str:
        if not value:
            return ""
        match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", value)
        if match is None:
            return value
        year, month_text = match.groups()
        if month_text is None:
            return year
        month = int(month_text)
        if not 1 <= month <= 12:
            return value
        month_name = _MONTH_NAMES[language][month - 1]
        return f"{month_name} {year}" if language == "en" else f"{month_name} de {year}"


class SynthesisFallbackRenderer:
    """Render a concise canonical subset when transformation cannot be delivered."""

    MAX_SENTENCES = MAX_SYNTHESIS_SENTENCES
    MAX_WORDS = MAX_SYNTHESIS_WORDS

    def __init__(self, profile: Profile) -> None:
        self._catalog = {
            fact.fact_id: fact for fact in build_resume_fact_catalog(profile)
        }

    def render(self, plan: AnswerPlan, *, max_words: int | None = None) -> str:
        word_limit = max_words or self.MAX_WORDS
        selected = [
            self._catalog[fact_id]
            for fact_id in plan.selected_fact_ids
            if fact_id in self._catalog
        ]
        if not selected:
            return ""
        # A fallback is intentionally a representative subset, never a narrative dump.
        parts: list[str] = []
        for fact in selected[:1]:
            candidate = fact_display_text(fact, plan.language).strip()
            if plan.topic == "projects" and fact.entity and fact.entity not in candidate:
                candidate = f"{fact.entity}: {candidate}"
            proposed = " ".join([*parts, candidate])
            if len(proposed.split()) > word_limit:
                continue
            if self._sentence_count(proposed) > self.MAX_SENTENCES:
                continue
            parts.append(candidate)
        if not parts:
            words = fact_display_text(selected[0], plan.language).split()
            return " ".join(words[:word_limit]).rstrip(".,;:") + "."
        return " ".join(parts)

    @staticmethod
    def _sentence_count(text: str) -> int:
        return max(1, len(re.findall(r"[.!?]+(?=\s+[A-ZÁÉÍÓÚÑ¿¡]|\s*$)", text)))


def plan_trace_fields(plan: AnswerPlan, rendering_mode: str) -> dict[str, object]:
    """Project an answer plan into the stable internal trace fields."""
    return {
        "answer_mode": plan.mode.value,
        "rendering_mode": rendering_mode,
        "answer_topic": plan.topic,
        "answer_scope": plan.scope,
        "requested_field": plan.requested_field,
        "synthesis_dimension": plan.synthesis_dimension,
        "selected_fact_ids": plan.selected_fact_ids,
        "selected_source_ids": plan.selected_source_ids,
    }
