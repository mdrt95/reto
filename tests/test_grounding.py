"""Focused tests for deterministic claim-source validation."""

import pytest

from src.agent.contracts import Claim, ClaimKind
from src.agent.grounding import verify_claims
from src.models.profile import load_profile


def test_grounding_accepts_claim_with_known_source() -> None:
    """A direct claim linked to an existing profile highlight is grounded."""
    profile = load_profile("data/profile.json")

    result = verify_claims(
        profile,
        [
            Claim(
                text="Built a hybrid retrieval pipeline combining FAISS semantic vector search with SQLite FTS5 full-text search.",
                kind=ClaimKind.DIRECT,
                source_ids=["project:proj-sybil.highlight:sybil-hl-hybrid"],
                evidence=["Built a hybrid retrieval pipeline combining FAISS semantic vector search with SQLite FTS5 full-text search."],
            )
        ],
    )

    assert result.status == "fully_grounded"
    assert result.claim_sources == {
        0: ["project:proj-sybil.highlight:sybil-hl-hybrid"]
    }


def test_grounding_rejects_unknown_source_id() -> None:
    """A plausible-looking claim cannot become grounded without a real source."""
    profile = load_profile("data/profile.json")

    result = verify_claims(
        profile,
        [
            Claim(
                text="Marco worked at Google.",
                kind=ClaimKind.DIRECT,
                source_ids=["experience:exp-google"],
                evidence=["Marco worked at Google."],
            )
        ],
    )

    assert result.status == "not_grounded"
    assert result.unsupported_claims == ["Marco worked at Google."]


def test_grounding_rejects_named_term_missing_from_a_valid_source() -> None:
    """A valid citation ID cannot be used to support an unrelated technology claim."""
    profile = load_profile("data/profile.json")

    result = verify_claims(
        profile,
        [
            Claim(
                text="Sybil uses AWS for retrieval.",
                kind=ClaimKind.DIRECT,
                source_ids=["project:proj-sybil.highlight:sybil-hl-hybrid"],
                evidence=["Built a hybrid retrieval pipeline combining FAISS semantic vector search with SQLite FTS5 full-text search."],
            )
        ],
    )

    assert result.status == "not_grounded"


def test_grounding_rejects_vague_claim_supported_only_by_a_person_name() -> None:
    """A person-name citation alone is not evidence for an unsupported attribute."""
    profile = load_profile("data/profile.json")

    result = verify_claims(
        profile,
        [
            Claim(
                text="Marco is successful.",
                kind=ClaimKind.DIRECT,
                source_ids=["personal"],
                evidence=["Software Engineer | AI/LLM Engineering"],
            )
        ],
    )

    assert result.status == "not_grounded"


@pytest.mark.parametrize(
    "contradictory_text",
    ["Marco worked at Google.", "Marco trabajó en Google."],
)
def test_selected_fact_id_does_not_ground_arbitrary_provider_prose(
    contradictory_text: str,
) -> None:
    """Fact identity authorizes selection, never semantic provider prose."""
    profile = load_profile("data/profile.json")
    claim = Claim(
        text=contradictory_text,
        kind=ClaimKind.DIRECT,
        fact_ids=["fact:experience:exp-global-payments"],
        source_ids=["experience:exp-global-payments"],
        evidence=[],
    )

    result = verify_claims(
        profile,
        [claim],
        selected_fact_ids={"fact:experience:exp-global-payments"},
    )

    assert result.status == "not_grounded"
    assert result.claim_sources == {}
    assert result.claim_fact_ids == {0: ["fact:experience:exp-global-payments"]}


def test_fact_id_validation_rejects_unselected_or_mismatched_sources() -> None:
    profile = load_profile("data/profile.json")
    claim = Claim(
        text="Unsupported translation",
        kind=ClaimKind.DIRECT,
        fact_ids=["fact:project:proj-sybil.highlight:sybil-hl-rag"],
        source_ids=["experience:exp-global-payments"],
        evidence=[],
    )

    result = verify_claims(
        profile,
        [claim],
        selected_fact_ids={"fact:project:proj-sybil.highlight:sybil-hl-rag"},
    )

    assert result.status == "not_grounded"


def test_legacy_inference_cannot_use_exact_evidence_to_authorize_new_prose() -> None:
    """Exact excerpts cannot prove an arbitrary synthesis without semantic judgment."""
    profile = load_profile("data/profile.json")
    claim = Claim(
        text="Marco worked at Google.",
        kind=ClaimKind.INFERRED,
        source_ids=["experience:exp-global-payments", "project:proj-sybil"],
        evidence=[
            "Jr. .NET Developer (Full-Stack)",
            "Python Retrieval-Augmented Document Q&A",
        ],
    )

    result = verify_claims(profile, [claim])

    assert result.status == "not_grounded"
