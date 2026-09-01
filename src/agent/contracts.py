"""Typed contracts crossing agent, tool, and verification boundaries."""

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Allowlisted conversation intents recognized by the orchestrator."""

    DIRECT_QUESTION = "direct_question"
    SEARCH_QUERY = "search_query"
    FILTER_REQUEST = "filter_request"
    SUMMARY_REQUEST = "summary_request"
    FOLLOW_UP = "follow_up"
    OUT_OF_SCOPE = "out_of_scope"
    ADVERSARIAL = "adversarial"


class IntentDecision(BaseModel):
    """Validated model output used to create a bounded tool plan."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    query: str | None = None
    filter_by: str | None = None
    filter_value: str | None = None
    audience: str | None = None
    profile_field: str | None = None


class ClaimKind(str, Enum):
    """Whether a generated statement is direct evidence or an inference."""

    DIRECT = "direct"
    INFERRED = "inferred"


class Claim(BaseModel):
    """A generated statement paired with its claimed profile evidence."""

    text: str = Field(min_length=1)
    kind: ClaimKind
    source_ids: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class GeneratedResponse(BaseModel):
    """Provider output before deterministic grounding and output checks."""

    text: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)


class GroundingResult(BaseModel):
    """Internal result of deterministic claim-to-source validation."""

    status: str
    claims_checked: int
    claims_grounded: int
    unsupported_claims: list[str] = Field(default_factory=list)
    claim_sources: dict[int, list[str]] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """Sanitized per-turn metadata used by logs, never public diagnostics."""

    guardrail_input: str = "pass"
    guardrail_output: str = "pass"
    intent: str | None = None
    intent_confidence: float | None = None
    tool_name: str | None = None
    tool_result_count: int = 0
    grounding_status: str | None = None
    claim_source_ids: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """The core service result exposed through delivery adapters."""

    answer: str
    trace: AgentTrace
