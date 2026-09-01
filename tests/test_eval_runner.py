"""Unit tests for the eval harness's whole-word token matcher."""

from eval.run_eval import token_present


def test_whole_word_token_matches_exact_phrase() -> None:
    """A required token must match when it appears as its own word/phrase."""
    assert token_present("Security Console", "Built the Security Console for provisioning.") is True


def test_whole_word_token_is_case_insensitive() -> None:
    """Token matching must not depend on the answer's casing."""
    assert token_present("security console", "Built the SECURITY CONSOLE for provisioning.") is True


def test_forbidden_token_does_not_match_inside_a_longer_word() -> None:
    """A substring match previously false-positived on a token embedded in another word."""
    assert token_present("led", "He scheduled the meeting.") is False


def test_forbidden_token_matches_as_its_own_word() -> None:
    """The same token must still match when it genuinely appears as a standalone word."""
    assert token_present("led", "He led the team.") is True
