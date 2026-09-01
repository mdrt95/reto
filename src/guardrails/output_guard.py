"""Deterministic output checks that prevent forbidden disclosures."""

from pydantic import BaseModel

from src.models.profile import Profile


class OutputGuardResult(BaseModel):
    """A safe decision on whether generated text may be delivered publicly."""

    allowed: bool
    code: str


def evaluate_output(
    text: str,
    profile: Profile,
    *,
    contact_requested: bool = False,
) -> OutputGuardResult:
    """Reject known sensitive or internal content before public delivery."""
    normalized = text.casefold()
    if profile.personal.phone and profile.personal.phone.casefold() in normalized:
        return OutputGuardResult(allowed=False, code="phone_disclosure")
    if profile.personal.email.casefold() in normalized and not contact_requested:
        return OutputGuardResult(allowed=False, code="email_disclosure")
    if "system prompt" in normalized or "developer message" in normalized:
        return OutputGuardResult(allowed=False, code="internal_instruction_leak")
    return OutputGuardResult(allowed=True, code="pass")
