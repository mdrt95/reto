"""Bounded orchestration that keeps model output behind typed policies."""

from typing import Protocol

from src.agent.contracts import AgentResponse, AgentTrace, GeneratedResponse, Intent, IntentDecision
from src.agent.grounding import profile_source_ids, verify_claims
from src.guardrails.input_guard import evaluate_input
from src.guardrails.output_guard import evaluate_output
from src.models.profile import Profile
from src.tools.profile_tools import (
    ExperienceFilterResult,
    ProfileQueryResult,
    ProfileSummaryPlan,
    ProjectSearchResult,
    FilterExperienceArguments,
    QueryProfileArguments,
    SearchProjectsArguments,
    SummarizeProfileArguments,
    filter_experience,
    query_profile,
    search_projects,
    summarize_profile,
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
        contact_requested: bool,
    ) -> GeneratedResponse: ...


ToolResult = ExperienceFilterResult | ProfileQueryResult | ProfileSummaryPlan | ProjectSearchResult


class AgentService:
    """Run one bounded, traceable profile-answering workflow per chat turn."""

    def __init__(self, *, profile: Profile, classifier: IntentClassifier, generator: ResponseGenerator) -> None:
        self._profile = profile
        self._classifier = classifier
        self._generator = generator

    def respond(self, message: str, *, history: list[object]) -> AgentResponse:
        """Answer safely or return a verified boundary response for this turn."""
        input_result = evaluate_input(message)
        if not input_result.allowed:
            return AgentResponse(
                answer=input_result.message,
                trace=AgentTrace(guardrail_input="blocked"),
            )

        decision = self._classifier.classify(message, history)
        if decision.intent in {Intent.OUT_OF_SCOPE, Intent.ADVERSARIAL}:
            return AgentResponse(
                answer="I'm focused on Marco's professional profile, experience, skills, and projects. What would you like to know?",
                trace=AgentTrace(
                    guardrail_input="blocked",
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                ),
            )
        if decision.confidence < 0.7:
            return AgentResponse(
                answer="Could you clarify which part of Marco's professional profile you mean?",
                trace=AgentTrace(intent=decision.intent.value, intent_confidence=decision.confidence),
            )

        tool_name, tool_result = self._execute_tool(decision)
        tool_result_count = len(getattr(tool_result, "matches", [])) if tool_result else 0
        allowed_sources = profile_source_ids(self._profile)
        contact_requested = self._is_explicit_contact_request(message)
        generated = self._generator.generate(
            message=message,
            history=history,
            profile=self._profile,
            tool_result=tool_result,
            allowed_source_ids=allowed_sources,
            contact_requested=contact_requested,
        )
        grounding = verify_claims(self._profile, generated.claims)
        if grounding.status != "fully_grounded":
            generated = self._generator.generate(
                message=message,
                history=history,
                profile=self._profile,
                tool_result=tool_result,
                allowed_source_ids=allowed_sources,
                contact_requested=contact_requested,
            )
            grounding = verify_claims(self._profile, generated.claims)
        if grounding.status != "fully_grounded":
            return AgentResponse(
                answer="I can only confirm information that is explicitly supported by Marco's profile.",
                trace=AgentTrace(
                    intent=decision.intent.value,
                    intent_confidence=decision.confidence,
                    tool_name=tool_name,
                    tool_result_count=tool_result_count,
                    grounding_status=grounding.status,
                ),
            )

        output_result = evaluate_output(
            generated.text,
            self._profile,
            contact_requested=contact_requested,
        )
        if not output_result.allowed:
            return AgentResponse(
                answer="I can help with Marco's public professional profile, but I can't provide that information.",
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
            answer=generated.text,
            trace=AgentTrace(
                intent=decision.intent.value,
                intent_confidence=decision.confidence,
                tool_name=tool_name,
                tool_result_count=tool_result_count,
                grounding_status=grounding.status,
                claim_source_ids=[
                    source_id for source_ids in grounding.claim_sources.values() for source_id in source_ids
                ],
            ),
        )

    def _execute_tool(self, decision: IntentDecision) -> tuple[str | None, ToolResult | None]:
        """Translate only validated intent fields into allowlisted read-only tools."""
        if decision.intent is Intent.SEARCH_QUERY:
            return "search_projects", search_projects(
                self._profile,
                SearchProjectsArguments(query=decision.query or "profile"),
            )
        if decision.intent is Intent.FILTER_REQUEST:
            filter_by = decision.filter_by if decision.filter_by in {"technology", "tag", "role"} else "tag"
            return "filter_experience", filter_experience(
                self._profile,
                FilterExperienceArguments(filter_by=filter_by, value=decision.filter_value or "profile"),
            )
        if decision.intent is Intent.SUMMARY_REQUEST:
            audience = decision.audience if decision.audience in {"technical", "recruiter", "executive"} else "recruiter"
            return "summarize_profile", summarize_profile(
                self._profile,
                SummarizeProfileArguments(audience=audience),
            )
        if decision.intent is Intent.DIRECT_QUESTION and decision.profile_field in {
            "skills",
            "languages",
            "education",
            "current_role",
        }:
            return "query_profile", query_profile(
                self._profile,
                QueryProfileArguments(field=decision.profile_field),
            )
        return None, None

    @staticmethod
    def _is_explicit_contact_request(message: str) -> bool:
        """Allow professional email only for an unmistakable request to contact Marco."""
        normalized = message.casefold()
        return "contact" in normalized or "reach marco" in normalized or "email marco" in normalized
