"""Focused tests for redacting source data before it crosses the model boundary."""

from src.agent.claude import profile_prompt_payload
from src.models.profile import load_profile


def test_model_payload_never_contains_phone_or_unsolicited_email() -> None:
    """The provider must not receive contact data unrelated to the current request."""
    profile = load_profile("data/profile.json")

    payload = profile_prompt_payload(profile, contact_requested=False)

    assert "phone" not in payload["personal"]
    assert "email" not in payload["personal"]
