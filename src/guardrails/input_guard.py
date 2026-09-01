"""Deterministic input guardrails that run before any model call."""

from pydantic import BaseModel


class InputGuardResult(BaseModel):
    """A safe decision on whether a request can enter orchestration."""

    allowed: bool
    code: str
    message: str


_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "reveal your system prompt",
    "show me your system prompt",
    "developer message",
)
_PHONE_PATTERNS = ("phone number", "telephone number", "cell number")
_OUT_OF_SCOPE_PATTERNS = ("what's the weather", "write me a poem")


def evaluate_input(message: str) -> InputGuardResult:
    """Block injections and sensitive probes without trusting client instructions."""
    normalized = message.casefold()
    if any(pattern in normalized for pattern in _INJECTION_PATTERNS):
        return InputGuardResult(
            allowed=False,
            code="prompt_injection",
            message="I can help only with Marco's professional profile, experience, skills, and projects.",
        )
    if any(pattern in normalized for pattern in _PHONE_PATTERNS):
        return InputGuardResult(
            allowed=False,
            code="pii_probe",
            message="I can share professional-profile information, but not private contact details.",
        )
    if any(pattern in normalized for pattern in _OUT_OF_SCOPE_PATTERNS):
        return InputGuardResult(
            allowed=False,
            code="out_of_scope",
            message="I'm focused on Marco's professional profile. What would you like to know about it?",
        )
    return InputGuardResult(allowed=True, code="pass", message="")
