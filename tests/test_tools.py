"""Focused behavioral tests for read-only profile tools."""

from src.models.profile import load_profile
from src.tools.profile_tools import (
    FilterExperienceArguments,
    QueryProfileArguments,
    SearchProjectsArguments,
    SummarizeProfileArguments,
    filter_experience,
    query_profile,
    search_projects,
    summarize_profile,
)


def test_search_projects_finds_faiss_with_stable_source_id() -> None:
    """Technology lookup must return sourceable, read-only project matches."""
    profile = load_profile("data/profile.json")

    result = search_projects(profile, SearchProjectsArguments(query="FAISS"))

    assert result.matches[0].source_id == "project:proj-sybil.highlight:sybil-hl-hybrid"
    assert "FAISS" in result.matches[0].summary


def test_filter_experience_finds_security_tag() -> None:
    """Experience filters must search normalized tags without model involvement."""
    profile = load_profile("data/profile.json")

    result = filter_experience(profile, FilterExperienceArguments(filter_by="tag", value="security"))

    assert {match.highlight_id for match in result.matches} == {
        "hl-security-console",
        "hl-performance",
    }


def test_query_profile_excludes_contact_data() -> None:
    """The public projection must never offer unrestricted contact fields."""
    profile = load_profile("data/profile.json")

    result = query_profile(profile, QueryProfileArguments(field="languages"))

    assert result.field == "languages"
    assert "phone" not in result.value
    assert "email" not in result.value


def test_query_profile_returns_employers_with_stable_source_ids() -> None:
    """Employment history must be exact, deduplicated, and sourceable."""
    profile = load_profile("data/profile.json")

    result = query_profile(profile, QueryProfileArguments(field="companies"))

    assert result.value == ["Global Payments (EVO Payments México)"]
    assert result.source_ids == ["experience:exp-global-payments"]


def test_query_profile_returns_education_with_stable_source_ids() -> None:
    profile = load_profile("data/profile.json")

    result = query_profile(profile, QueryProfileArguments(field="education"))

    assert result.value == ["B.S. in ICT Engineering — Instituto Tecnológico de Ciudad Madero (ITCM)"]
    assert result.source_ids == ["education:edu-itcm-ict"]


def test_summary_tool_returns_verified_sources_without_generating_copy() -> None:
    """Summary planning selects sources; the model writes the prose later."""
    profile = load_profile("data/profile.json")

    result = summarize_profile(profile, SummarizeProfileArguments(audience="technical"))

    assert result.audience == "technical"
    assert "project:proj-sybil" in result.source_ids
