"""Deterministic containment-gate tests for model-rephrased answers (D-029)."""

import pytest

from src.agent.rephrase import verify_rephrase, verify_synthesis_text
from src.models.profile import load_profile
from src.tools.profile_tools import ResumeFact, build_resume_fact_catalog


@pytest.fixture()
def catalog() -> list[ResumeFact]:
    return build_resume_fact_catalog(load_profile("data/profile.json"))


def _fact(catalog: list[ResumeFact], source_id: str) -> ResumeFact:
    for fact in catalog:
        if fact.source_id == source_id:
            return fact
    raise AssertionError(f"fact not found for source_id={source_id}")


def test_faithful_english_paraphrase_is_accepted(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text=(
            "Marco developed an internal Security Console for provisioning users, "
            "roles, and permissions across onboarding applications."
        ),
        selected_facts=[security_console],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is True
    assert verdict.code == "accepted"


def test_role_escalation_is_rejected(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text="Marco led the team that built the Security Console.",
        selected_facts=[security_console],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is False
    assert verdict.code == "escalation"


def test_managed_verb_is_rejected_as_escalation(catalog: list[ResumeFact]) -> None:
    performance = _fact(catalog, "experience:exp-global-payments.highlight:hl-performance")

    verdict = verify_rephrase(
        text="Marco managed the caching and rate limiting rollout.",
        selected_facts=[performance],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is False
    assert verdict.code == "escalation"


def test_foreign_vocabulary_not_in_profile_is_rejected(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text="Marco built this feature while working with Google engineers.",
        selected_facts=[security_console],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is False
    assert verdict.code == "foreign_vocabulary"


def test_leaked_fact_from_elsewhere_in_profile_is_rejected(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text="Marco built this internal system integrated with FAISS for search.",
        selected_facts=[security_console],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is False
    assert verdict.code == "leaked_fact"


def test_alternate_verb_form_is_accepted(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text="Marco developed the internal Security Console.",
        selected_facts=[security_console],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is True


def test_faithful_spanish_paraphrase_using_construyo_is_accepted(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text=(
            "Marco construyó una Security Console interna para aprovisionar usuarios, "
            "roles y permisos."
        ),
        selected_facts=[security_console],
        catalog=catalog,
        language="es",
    )

    assert verdict.allowed is True
    assert verdict.code == "accepted"


def test_three_sentences_for_one_fact_is_too_long(catalog: list[ResumeFact]) -> None:
    security_console = _fact(catalog, "experience:exp-global-payments.highlight:hl-security-console")

    verdict = verify_rephrase(
        text="Marco built the Security Console. It handles roles. It handles permissions.",
        selected_facts=[security_console],
        catalog=catalog,
        language="en",
    )

    assert verdict.allowed is False
    assert verdict.code == "too_long"


def test_length_check_ignores_abbreviation_dots_and_budgets_by_narrative_length() -> None:
    """A faithful restatement of long Spanish narratives must not fail the length gate."""
    from src.agent.rephrase import verify_rephrase
    from src.models.profile import load_profile
    from src.tools.profile_tools import build_resume_fact_catalog, fact_display_text

    profile = load_profile("data/profile.json")
    catalog = build_resume_fact_catalog(profile)
    selected = [fact for fact in catalog if fact.source_id.startswith("experience:exp-global-payments")][:6]
    text = " ".join(fact_display_text(fact, "es") for fact in selected)
    assert "Jr. .NET" in text

    verdict = verify_rephrase(text=text, selected_facts=selected, catalog=catalog, language="es")

    assert verdict.code == "accepted", verdict.details


def test_sentence_count_ignores_abbreviation_and_dotted_token_boundaries() -> None:
    """'Jr. .NET' and 'Node.js' must not inflate the sentence count used by the length gate."""
    from src.agent.rephrase import count_sentences

    assert count_sentences("Marco trabaja como Jr. .NET Developer en Global Payments. Marco construyó Node.js APIs.") == 2
    assert count_sentences("Marco built it. He shipped it! Did it work? Yes.") == 4


def test_word_budget_scales_with_selected_narrative_length() -> None:
    """A restatement slightly longer than long Spanish narratives must still pass."""
    from src.agent.rephrase import verify_rephrase
    from src.models.profile import load_profile
    from src.tools.profile_tools import build_resume_fact_catalog, fact_display_text

    profile = load_profile("data/profile.json")
    catalog = build_resume_fact_catalog(profile)
    selected = [fact for fact in catalog if fact.source_id.startswith("experience:exp-global-payments")][:6]
    text = " ".join(fact_display_text(fact, "es") for fact in selected) + " Marco también apoya a su equipo con esas tareas cada día."
    assert len(text.split()) > 40 * len(selected) + 20

    verdict = verify_rephrase(text=text, selected_facts=selected, catalog=catalog, language="es")

    assert verdict.code == "accepted", verdict.details


def test_numbers_with_thousand_separators_match_profile_vocabulary() -> None:
    """'1,024-dimensional' appears in the profile and must not be flagged as foreign."""
    from src.agent.rephrase import verify_rephrase
    from src.models.profile import load_profile
    from src.tools.profile_tools import build_resume_fact_catalog

    profile = load_profile("data/profile.json")
    catalog = build_resume_fact_catalog(profile)
    selected = [fact for fact in catalog if fact.fact_id.endswith("sybil-hl-hybrid")]
    text = (
        "Marco built a hybrid retrieval pipeline combining FAISS semantic vector search with "
        "SQLite FTS5 full-text search, using 1,024-dimensional Voyage AI voyage-3-lite embeddings."
    )

    verdict = verify_rephrase(text=text, selected_facts=selected, catalog=catalog, language="en")

    assert verdict.code == "accepted", verdict.details


def test_a_verb_the_cited_fact_itself_uses_is_not_drift(catalog: list[ResumeFact]) -> None:
    """The fact's leading verb is not the only verb it authorizes.

    `hl-isv-module` reads "Collaborated in delivering ... Built an internal multi-agent
    engineering workflow". Rejecting "built" rejects the fact's own wording.
    """
    isv_module = _fact(catalog, "experience:exp-global-payments.highlight:hl-isv-module")

    verdict = verify_rephrase(
        text=(
            "Marco collaborated in delivering a new ISV module, beating deadline "
            "expectations, by building an internal multi-agent engineering workflow."
        ),
        selected_facts=[isv_module],
        catalog=catalog,
        language="en",
        require_each_fact_verb=False,
    )

    assert verdict.allowed, verdict.details


def test_synthesis_admits_inflections_of_authorized_words(catalog: list[ResumeFact]) -> None:
    """Compression rewrites grammar; "implemented" becoming "implementing" adds no fact."""
    performance = _fact(catalog, "experience:exp-global-payments.highlight:hl-performance")

    verdict = verify_synthesis_text(
        text=(
            "Marco resolved availability-affecting performance bottlenecks by "
            "implementing Redis and SQL caching per Global Payments requirements."
        ),
        selected_facts=[performance],
        catalog=catalog,
        language="en",
        dimension="summary",
    )

    assert verdict.allowed, verdict.details


def test_synthesis_still_rejects_a_word_absent_from_the_selection(
    catalog: list[ResumeFact],
) -> None:
    """Inflection tolerance must not become an opening for unrelated content."""
    performance = _fact(catalog, "experience:exp-global-payments.highlight:hl-performance")

    verdict = verify_synthesis_text(
        text="Marco implemented caching, delighting every enterprise stakeholder.",
        selected_facts=[performance],
        catalog=catalog,
        language="en",
        dimension="summary",
    )

    assert not verdict.allowed
    assert verdict.code == "unsupported_vocabulary"


def test_a_possessive_s_is_tokenization_residue_not_a_claim(
    catalog: list[ResumeFact],
) -> None:
    """`normalize_resume_text` splits "Marco's" into "marco" + "s"."""
    sybil = _fact(catalog, "project:proj-sybil")

    verdict = verify_synthesis_text(
        text="Sybil is Marco's Retrieval-Augmented Generation system.",
        selected_facts=[sybil],
        catalog=catalog,
        language="en",
        dimension="summary",
    )

    assert verdict.allowed, verdict.details
