"""Deterministic validation of generated claims against stable profile IDs."""

import re

from src.agent.contracts import Claim, ClaimKind, GroundingResult
from src.models.profile import Profile
from src.tools.profile_tools import build_resume_fact_catalog


def profile_source_ids(profile: Profile) -> set[str]:
    """Build the stable identifier catalog accepted by grounding verification."""
    source_ids = {"personal", "skills"}
    if profile.professional_summary:
        source_ids.add("professional_summary")
    if profile.career_preferences:
        source_ids.add("career_preferences")
    for experience in profile.experience:
        source_ids.add(f"experience:{experience.id}")
        source_ids.update(
            f"experience:{experience.id}.highlight:{highlight.id}"
            for highlight in experience.highlights
        )
    for project in profile.projects:
        source_ids.add(f"project:{project.id}")
        source_ids.update(
            f"project:{project.id}.highlight:{highlight.id}"
            for highlight in project.highlights
        )
    source_ids.update(f"education:{item.id}" for item in profile.education)
    return source_ids


def _source_text_catalog(profile: Profile) -> dict[str, str]:
    """Associate every stable source ID with the profile text it can support."""
    catalog = {
        "personal": " ".join(
            [
                profile.personal.name,
                profile.personal.title,
                profile.personal.location,
                *[f"{item.language} {item.level}" for item in profile.personal.languages],
            ]
        ),
        "skills": " ".join(
            [
                *profile.skills.programming_languages,
                *profile.skills.ai_llm,
                *profile.skills.ai_stack,
                *profile.skills.backend_apis,
                *profile.skills.devops_engineering,
            ]
        ),
    }
    for experience in profile.experience:
        experience_text = " ".join(
            [profile.personal.name, experience.role, experience.company, experience.team_context]
        )
        catalog[f"experience:{experience.id}"] = experience_text
        for highlight in experience.highlights:
            catalog[f"experience:{experience.id}.highlight:{highlight.id}"] = " ".join(
                [experience_text, highlight.summary, highlight.detail, *highlight.technologies, *highlight.tags]
            )
    for project in profile.projects:
        project_text = " ".join(
            [profile.personal.name, project.name, project.subtitle, *project.technologies]
        )
        catalog[f"project:{project.id}"] = project_text
        for highlight in project.highlights:
            catalog[f"project:{project.id}.highlight:{highlight.id}"] = " ".join(
                [project_text, highlight.summary, highlight.detail, *highlight.technologies, *highlight.tags]
            )
    for education in profile.education:
        catalog[f"education:{education.id}"] = " ".join(
            [education.degree, education.institution, str(education.start_year), str(education.end_year)]
        )
    if profile.professional_summary:
        catalog["professional_summary"] = profile.professional_summary
    if profile.career_preferences:
        catalog["career_preferences"] = " ".join(
            [
                *profile.career_preferences.desired_roles,
                profile.career_preferences.seniority or "",
                *profile.career_preferences.locations,
                *profile.career_preferences.work_arrangements,
                *profile.career_preferences.notes,
            ]
        )
    return catalog


def _normalize_evidence(text: str) -> str:
    """Normalize profile text for exact, punctuation-insensitive evidence matching."""
    return re.sub(r"[^a-z0-9#+]+", "", text.casefold())


def _has_deterministic_evidence(claim: Claim, source_text: str) -> bool:
    """Approve only exact direct legacy prose; synthesized prose is not provable."""
    if claim.kind is not ClaimKind.DIRECT or not claim.evidence:
        return False
    normalized_source = _normalize_evidence(source_text)
    evidence_is_exact = all(
        _normalize_evidence(excerpt) in normalized_source for excerpt in claim.evidence
    )
    if not evidence_is_exact:
        return False
    return evidence_is_exact and _normalize_evidence(claim.text) in normalized_source


def verify_claims(
    profile: Profile,
    claims: list[Claim],
    *,
    selected_fact_ids: set[str] | None = None,
) -> GroundingResult:
    """Separate exact grounded prose from authorized canonical fact selections."""
    known_sources = profile_source_ids(profile)
    source_catalog = _source_text_catalog(profile)
    fact_catalog = {fact.fact_id: fact for fact in build_resume_fact_catalog(profile)}
    claim_sources: dict[int, list[str]] = {}
    claim_fact_ids: dict[int, list[str]] = {}
    unsupported_claims: list[str] = []

    for index, claim in enumerate(claims):
        sources_are_known = all(source_id in known_sources for source_id in claim.source_ids)
        cited_text = " ".join(source_catalog.get(source_id, "") for source_id in claim.source_ids)
        facts_are_selected = bool(claim.fact_ids) and all(
            fact_id in fact_catalog
            and (selected_fact_ids is None or fact_id in selected_fact_ids)
            for fact_id in claim.fact_ids
        )
        fact_sources_match = facts_are_selected and all(
            fact_catalog[fact_id].source_id in claim.source_ids for fact_id in claim.fact_ids
        )
        if sources_are_known and fact_sources_match:
            # Fact IDs authorize only canonical fact selection. They cannot prove
            # arbitrary provider prose semantically matches those facts.
            claim_fact_ids[index] = claim.fact_ids
            unsupported_claims.append(claim.text)
        elif sources_are_known and not claim.fact_ids and _has_deterministic_evidence(claim, cited_text):
            # Compatibility path for existing exact-English provider/test payloads.
            claim_sources[index] = claim.source_ids
        else:
            unsupported_claims.append(claim.text)

    checked = len(claims)
    grounded = len(claim_sources)
    status = (
        "fully_grounded"
        if grounded == checked
        else "not_grounded"
        if grounded == 0
        else "partially_grounded"
    )
    return GroundingResult(
        status=status,
        claims_checked=checked,
        claims_grounded=grounded,
        unsupported_claims=unsupported_claims,
        claim_sources=claim_sources,
        claim_fact_ids=claim_fact_ids,
    )
