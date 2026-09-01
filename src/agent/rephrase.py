"""Deterministic containment gate for model-rephrased answers (see DECISIONS.md D-029).

The gate never judges truth — the orchestrator already restricted the model to the turn's
selected facts before generation. This module only checks that the rephrase stayed inside
that selection's vocabulary, verb meaning, and length; any failure means the caller must
fall back to the canonical (A) rendering instead of delivering the model's prose.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.tools.profile_tools import fact_display_text, ResumeFact, normalize_resume_text

_ESCALATION_TOKENS = {
    "led", "lead", "leads", "leading", "managed", "manage", "manages",
    "owned", "owner", "architected", "founded", "senior", "head",
    "principal", "director", "chief",
    "lidero", "lidera", "dirigio", "dirige", "gestiono", "gestiona",
    "encabezo", "fundo", "arquitecto", "jefe", "lider", "responsable",
}

_VERB_MAP: dict[str, set[str]] = {
    "built": {
        "built", "build", "builds", "building", "developed", "develop",
        "created", "create", "construyo", "construye", "construir",
        "desarrollo", "desarrolla", "desarrollar", "creo", "crea", "crear",
    },
    "implemented": {
        "implemented", "implement", "implements", "added", "add",
        "implemento", "implementa", "implementar", "agrego", "agrega",
        "agregar", "incorporo",
    },
    "assisted": {
        "assisted", "assist", "assists", "supported", "support", "helped",
        "help", "apoyo", "apoya", "apoyar", "asistio", "asiste", "ayudo",
        "ayuda", "ayudar",
    },
    "collaborated": {
        "collaborated", "collaborate", "contributed", "contribute",
        "colaboro", "colabora", "colaborar", "contribuyo", "contribuye",
        "participo", "participa",
    },
    "integrated": {
        "integrated", "integrate", "integrates", "integro", "integra", "integrar",
    },
}

# The profile's owner is always referenceable regardless of which facts were selected.
_NAME_TOKENS = {"marco", "reyes"}

# Split only on sentence punctuation; a dot inside a token (Node.js, ASP.NET) is not a boundary.
_SENTENCE_SPLIT_RE = re.compile(r"[?!¿¡]+|\.(?=\s|$)")
# A sentence ends only where terminal punctuation is followed by a capitalized word or the end,
# so abbreviations ("Jr.") and dotted product names (".NET", "Node.js") are not boundaries.
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s+[A-ZÁÉÍÓÚÑ¿¡]|\s*$)")


def count_sentences(text: str) -> int:
    """Count sentences for the length gate without treating abbreviation dots as boundaries."""
    stripped = text.strip()
    if not stripped:
        return 0
    ends = len(_SENTENCE_END_RE.findall(stripped))
    return max(1, ends if _SENTENCE_END_RE.search(stripped[-1:]) or stripped[-1] in ".!?" else ends + 1)
_ENTITY_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9][A-Za-zÁÉÍÓÚÑáéíóúñ0-9#+.,]*")


class RephraseVerdict(BaseModel):
    """Deterministic accept/reject decision for one candidate rephrase."""

    allowed: bool
    code: str
    details: list[str] = Field(default_factory=list)


def _fact_vocabulary(fact: ResumeFact) -> set[str]:
    """Build the normalized-token vocabulary a fact authorizes for a rephrase.

    `normalize_resume_text` keeps a trailing period stuck to its token (e.g. "jr."),
    while an entity-like candidate token strips trailing punctuation before comparison
    (matching `_profile_known_tokens`'s handling of dotted names such as "asp.net").
    Adding the stripped form here keeps both sides of the comparison consistent.
    """
    parts = [fact.text, *fact.keywords]
    if fact.entity:
        parts.append(fact.entity)
    if fact.narrative_en:
        parts.append(fact.narrative_en)
    if fact.narrative_es:
        parts.append(fact.narrative_es)
    joined = " ".join(parts)
    tokens = set(normalize_resume_text(joined).split())
    # Numbers with thousand separators are compared without the comma (1,024 -> 1024).
    tokens |= set(normalize_resume_text(joined.replace(",", "")).split())
    tokens |= {token.rstrip(".") for token in tokens if token.endswith(".")}
    return tokens


def _normalize_entity_token(raw: str) -> str:
    """Normalize one raw entity-like token, collapsing thousand separators (1,024 -> 1024)."""
    return normalize_resume_text(raw.replace(",", ""))


def _entity_like_tokens(text: str) -> list[str]:
    """Return raw entity-like tokens in appearance order, skipping sentence-initial words.

    Entity-like: capitalized (and not the first word of its sentence), contains a digit,
    or contains one of ".#+" — the signals a proper noun, product name, or number carries
    that ordinary prose does not.
    """
    tokens: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        matches = list(_ENTITY_TOKEN_RE.finditer(sentence))
        if not matches:
            continue
        sentence_start = matches[0].start()
        for match in matches:
            raw = match.group(0).rstrip(".,")
            if not raw:
                continue
            if match.start() == sentence_start:
                continue
            has_digit = any(character.isdigit() for character in raw)
            has_symbol = any(character in raw for character in ".#+")
            is_capitalized = raw[0].isupper()
            if has_digit or has_symbol or is_capitalized:
                tokens.append(raw)
    return tokens


def verify_rephrase(
    *,
    text: str,
    selected_facts: list[ResumeFact],
    catalog: list[ResumeFact],
    language: Literal["en", "es"],
) -> RephraseVerdict:
    """Deterministically decide whether model-rephrased prose stays inside selected facts.

    Checks run in order and the first failure wins: empty, escalation vocabulary, foreign
    or leaked vocabulary, verb-frame drift, then length. `language` selects the display text used for the length budget
    (every vocabulary check is language-normalized already).
    """
    if not text or not text.strip():
        return RephraseVerdict(allowed=False, code="empty", details=["Rephrase text is blank"])

    selected_fact_ids = {fact.fact_id for fact in selected_facts}
    selected_vocabulary: set[str] = set()
    for fact in selected_facts:
        selected_vocabulary |= _fact_vocabulary(fact)

    normalized_text_tokens = normalize_resume_text(text).split()
    normalized_token_set = set(normalized_text_tokens)

    escalated = sorted((normalized_token_set & _ESCALATION_TOKENS) - selected_vocabulary)
    if escalated:
        return RephraseVerdict(
            allowed=False,
            code="escalation",
            details=[f"Escalation vocabulary not present in selected facts: {', '.join(escalated)}"],
        )

    other_fact_vocabulary: list[set[str]] = [
        _fact_vocabulary(fact) for fact in catalog if fact.fact_id not in selected_fact_ids
    ]

    for raw_token in _entity_like_tokens(text):
        normalized_token = _normalize_entity_token(raw_token)
        if not normalized_token or normalized_token in _NAME_TOKENS:
            continue
        if normalized_token in selected_vocabulary:
            continue
        found_elsewhere = any(normalized_token in vocabulary for vocabulary in other_fact_vocabulary)
        if found_elsewhere:
            return RephraseVerdict(
                allowed=False,
                code="leaked_fact",
                details=[f"'{raw_token}' belongs to a fact outside this turn's selection"],
            )
        return RephraseVerdict(
            allowed=False,
            code="foreign_vocabulary",
            details=[f"'{raw_token}' is not present anywhere in the profile"],
        )

    for fact in selected_facts:
        fact_words = fact.text.split()
        if not fact_words:
            continue
        first_token = normalize_resume_text(fact_words[0])
        allowed_forms = _VERB_MAP.get(first_token)
        if allowed_forms is None:
            continue
        if not (normalized_token_set & allowed_forms):
            return RephraseVerdict(
                allowed=False,
                code="verb_frame",
                details=[f"Missing an allowed verb form for '{first_token}'"],
            )

    sentence_count = count_sentences(text)
    word_count = len(normalized_text_tokens)
    # Budget scales with the canonical display text so long narratives are not penalized.
    word_budget = 20 + sum(
        max(40, int(1.3 * len(fact_display_text(fact, language).split()))) for fact in selected_facts
    )
    if sentence_count > len(selected_facts) + 1 or word_count > word_budget:
        return RephraseVerdict(
            allowed=False,
            code="too_long",
            details=[f"{sentence_count} sentences / {word_count} words for {len(selected_facts)} facts"],
        )

    return RephraseVerdict(allowed=True, code="accepted", details=[])
