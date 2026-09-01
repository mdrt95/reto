"""Focused test for content-free structured observability events."""

import json

from src.observability.logger import TurnLogEvent


def test_turn_event_serializes_operational_metadata_without_content_fields() -> None:
    """Logs must support diagnosis without storing prompts, answers, or contact data."""
    event = TurnLogEvent(
        request_id="req_123",
        conversation_id="conv_123",
        outcome_code="completed",
        intent="search_query",
        tool_name="search_projects",
        tool_result_count=1,
        guardrail_input="pass",
        guardrail_output="pass",
        grounding_status="fully_grounded",
        latency_total_ms=42,
    )

    payload = json.loads(event.model_dump_json())

    assert payload["request_id"] == "req_123"
    assert "message" not in payload
    assert "answer" not in payload
    assert "email" not in payload
