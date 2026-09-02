"""Typed contracts crossing agent, tool, and verification boundaries."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class GenerationUnavailableError(RuntimeError):
    """Raised when model access or its required structured output is unavailable."""


class InvalidStructuredOutputError(GenerationUnavailableError):
    """Raised only when a provider response fails local structured validation."""


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


class AnswerMode(str, Enum):
    """Whether selected facts are rendered directly or transformed by synthesis."""

    DIRECT = "direct"
    SYNTHESIS = "synthesis"


AnswerTopic = Literal[
    "experience",
    "projects",
    "skills",
    "education",
    "languages",
    "summary",
    "career_preferences",
]

AnswerScope = Literal[
    "profile",
    "employment",
    "project",
    "skill",
    "education",
    "language",
    "career_preferences",
]

RequestedField = Literal[
    "start_date",
    "end_date",
    "current",
    "projects",
    "experience",
    "technology",
    "tag",
    "employer",
    "skills",
    "education",
    "languages",
    "current_role",
    "summary",
    "career_preferences",
]

MAX_SYNTHESIS_FACTS = 3
"""Evidence breadth one synthesis answer may draw on."""

MAX_SYNTHESIS_PROPOSITIONS = 2
"""Delivery cap shared by the provider contract and validation.

Deliberately below `MAX_SYNTHESIS_FACTS`: a full selection cannot be expressed as one
proposition per fact, so aggregation is structural rather than a prompt preference, and
the one-sentence-per-fact dump becomes unreachable by construction.
"""

MAX_SYNTHESIS_SENTENCES = 3
"""Default sentence budget for a synthesized answer and its canonical fallback."""

MAX_SYNTHESIS_WORDS = 75
"""Default word budget for a synthesized answer and its canonical fallback."""


SynthesisDimension = Literal[
    "summary",
    "impact",
    "significance",
    "comparison",
    "explanation",
    "conclusion",
]


class AnswerPlan(BaseModel):
    """Internal, typed selection contract resolved before answer rendering."""

    mode: AnswerMode
    topic: AnswerTopic
    scope: AnswerScope
    requested_field: RequestedField
    language: Literal["en", "es"]
    synthesis_dimension: SynthesisDimension | None = None
    selected_fact_ids: list[str] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)


class ClaimKind(str, Enum):
    """Whether a generated statement is direct evidence or an inference."""

    DIRECT = "direct"
    INFERRED = "inferred"


class Claim(BaseModel):
    """A generated statement paired with its claimed profile evidence."""

    text: str = Field(min_length=1)
    kind: ClaimKind
    fact_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class GeneratedResponse(BaseModel):
    """Provider output before deterministic grounding and output checks."""

    text: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)


class SynthesisProposition(BaseModel):
    """One transformed factual proposition mapped to selected canonical facts."""

    text: str = Field(min_length=1)
    fact_ids: list[str] = Field(min_length=1)


class SynthesisTransformation(BaseModel):
    """Structured transformation output before deterministic containment checks.

    The answer is carried only by `propositions`. A separate top-level copy of the
    same prose cannot add a guarantee — delivery already joins the mapped
    propositions — but it can drift from them and reject an otherwise valid answer.
    """

    propositions: list[SynthesisProposition] = Field(
        min_length=1, max_length=MAX_SYNTHESIS_PROPOSITIONS
    )


class GroundingResult(BaseModel):
    """Internal result of deterministic claim-to-source validation."""

    status: str
    claims_checked: int
    claims_grounded: int
    unsupported_claims: list[str] = Field(default_factory=list)
    claim_sources: dict[int, list[str]] = Field(default_factory=dict)
    claim_fact_ids: dict[int, list[str]] = Field(default_factory=dict)


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
    rephrase_outcome: str | None = None
    fallback_reason: str | None = None
    generator_skipped: bool = False
    answer_mode: str | None = None
    rendering_mode: str | None = None
    answer_topic: str | None = None
    answer_scope: str | None = None
    requested_field: str | None = None
    selected_fact_ids: list[str] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)
    synthesis_dimension: str | None = None
    transformation_outcome: str | None = None
    final_word_count: int | None = None
    final_sentence_count: int | None = None


StateValue = Annotated[str, Field(min_length=1, max_length=200)]


class ConversationState(BaseModel):
    """Compact client-carried state containing verified referents only."""

    last_topic: AnswerTopic | None = None
    last_source_ids: list[StateValue] = Field(default_factory=list, max_length=20)
    last_entities: list[StateValue] = Field(default_factory=list, max_length=8)
    last_tool: StateValue | None = None
    response_language: Literal["en", "es"] = "en"


class AgentResponse(BaseModel):
    """The core service result exposed through delivery adapters."""

    answer: str
    trace: AgentTrace
    state: ConversationState | None = None
