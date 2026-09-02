"""JSON event logging that deliberately excludes user and model content."""

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TurnLogEvent(BaseModel):
    """Operational metadata sufficient for quality and latency diagnosis."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str
    conversation_id: str
    route: str = "/api/chat"
    outcome_code: str
    intent: str | None = None
    intent_confidence: float | None = None
    tool_name: str | None = None
    tool_result_count: int = 0
    guardrail_input: str
    guardrail_output: str
    grounding_status: str | None = None
    rephrase_outcome: str | None = None
    fallback_reason: str | None = None
    generator_skipped: bool = False
    selection_path: str | None = None
    referent_source: str | None = None
    referent_correction: bool = False
    answer_mode: str | None = None
    rendering_mode: str | None = None
    synthesis_dimension: str | None = None
    transformation_outcome: str | None = None
    final_word_count: int | None = None
    final_sentence_count: int | None = None
    selected_fact_count: int = 0
    selected_source_count: int = 0
    claim_source_count: int = 0
    latency_total_ms: int
    latency_intent_ms: int | None = None
    latency_tool_ms: int | None = None
    latency_generation_ms: int | None = None
    latency_grounding_ms: int | None = None
    model_name: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_estimate_usd: float | None = None
    error_code: str | None = None


def configure_logging(level: str) -> None:
    """Configure stdout logging once with no request-content formatter."""
    logging.basicConfig(level=level.upper(), format="%(message)s")


def log_turn(event: TurnLogEvent) -> None:
    """Write one JSON event without prompts, answers, provider payloads, or PII."""
    logging.getLogger("banorte_cv_agent.turn").info(event.model_dump_json())
