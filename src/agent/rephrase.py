"""Deterministic containment gate for model-rephrased answers.

The gate never judges truth — the orchestrator already restricted the model to the turn's
selected facts before generation. This module only checks that the rephrase stayed inside
that selection's vocabulary, verb meaning, and length; any failure means the caller must
fall back to the canonical (A) rendering instead of delivering the model's prose.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.contracts import (
    MAX_SYNTHESIS_SENTENCES,
    MAX_SYNTHESIS_WORDS,
    SynthesisDimension,
)
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
_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?]+(?=\s|$)")
_ABBREVIATIONS = {"jr", "sr", "mr", "mrs", "ms", "dr", "prof", "e.g", "i.e"}


def count_sentences(text: str) -> int:
    """Count terminal punctuation regardless of next-sentence capitalization."""
    stripped = text.strip()
    if not stripped:
        return 0
    boundaries = 0
    last_boundary_end = 0
    for match in _SENTENCE_PUNCTUATION_RE.finditer(stripped):
        punctuation = match.group(0)
        if punctuation == ".":
            prefix = stripped[: match.start()]
            token_match = re.search(r"([A-Za-z.]+)$", prefix)
            preceding = token_match.group(1).casefold() if token_match else ""
            following = stripped[match.end():].lstrip()
            if preceding in _ABBREVIATIONS or following.startswith("."):
                continue
        boundaries += 1
        last_boundary_end = match.end()
    trailing_text = bool(stripped[last_boundary_end:].strip())
    return max(1, boundaries + (1 if trailing_text else 0))
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
    require_each_fact_verb: bool = True,
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
    source_verb_categories: set[str] = set()
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
        if require_each_fact_verb and not (normalized_token_set & allowed_forms):
            return RephraseVerdict(
                allowed=False,
                code="verb_frame",
                details=[f"Missing an allowed verb form for '{first_token}'"],
            )

    # A fact authorizes every verb meaning its own canonical wording uses, not just its
    # leading verb: "Collaborated in delivering ... Built an internal workflow" authorizes
    # "built". Escalation beyond the selection stays blocked by the escalation check above.
    source_verb_categories = {
        category
        for category, forms in _VERB_MAP.items()
        if selected_vocabulary & forms
    }
    used_verb_categories = {
        category
        for category, forms in _VERB_MAP.items()
        if normalized_token_set & forms
    }
    unsupported_verbs = sorted(used_verb_categories - source_verb_categories)
    if unsupported_verbs:
        return RephraseVerdict(
            allowed=False,
            code="verb_drift",
            details=[f"Verb meaning not present in selected facts: {', '.join(unsupported_verbs)}"],
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


_UNSUPPORTED_OUTCOME_CONCEPTS: dict[str, set[str]] = {
    "revenue": {"revenue", "revenues", "ingresos", "facturacion"},
    "profit": {"profit", "profits", "profitability", "ganancia", "ganancias"},
    "sales": {"sale", "sales", "ventas"},
    "savings": {"saving", "savings", "ahorro", "ahorros"},
    "growth": {"growth", "crecimiento"},
    "adoption": {"adoption", "adopcion"},
}

_LANGUAGE_MARKERS = {
    "en": {"built", "works", "worked", "implemented", "collaborated", "with", "and", "at"},
    "es": {"construyo", "trabaja", "trabajo", "implemento", "colaboro", "con", "y", "en"},
}

_FUNCTION_TOKENS = {
    "a", "an", "the", "and", "or", "but", "as", "at", "by", "for", "from", "in",
    "into", "on", "of", "to", "with", "that", "which", "who", "while", "through",
    "across", "about", "over", "under", "his", "her", "their", "its", "this", "these",
    "those", "it", "he", "she", "they", "was", "were", "is", "are", "be", "been",
    "being", "has", "have", "had", "also", "one", "example", "including", "using", "via",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "e", "o", "pero",
    "como", "en", "de", "del", "al", "por", "para", "con", "sin", "que", "quien",
    "su", "sus", "este", "esta", "estos", "estas", "lo", "se", "ha", "han", "fue",
    "es", "son", "siendo", "mediante", "desde", "hasta", "sobre",
    "work", "works", "worked", "working",
    "trabajo", "trabaja", "trabajado", "trabajar",
    "entre", "tambien", "asi", "muy", "toda", "todo", "todos", "todas",
    "s", "per", "each", "both", "such", "than", "then", "not", "no", "more", "most",
    "other", "another", "after", "before", "during", "within", "where", "when",
    "cada", "ambos", "tanto", "ademas", "antes", "despues", "durante", "dentro",
    "donde", "cuando", "mas", "otro", "otra", "otros", "otras",
}


def _language_vocabulary(
    facts: list[ResumeFact],
    language: Literal["en", "es"],
) -> set[str]:
    """Build language-specific evidence plus language-neutral entities/keywords."""
    values: list[str] = []
    for fact in facts:
        narrative = fact.narrative_en if language == "en" else fact.narrative_es
        if narrative:
            values.append(narrative)
        else:
            values.append(fact.text)
        values.extend(fact.keywords)
        if fact.entity:
            values.append(fact.entity)
    return {token.rstrip(".") for token in normalize_resume_text(" ".join(values)).split()}


def _authorized_verb_forms(facts: list[ResumeFact]) -> set[str]:
    forms: set[str] = set()
    for fact in facts:
        fact_words = fact.text.split()
        if not fact_words:
            continue
        category = normalize_resume_text(fact_words[0])
        forms.update(_VERB_MAP.get(category, set()))
    return forms


# Compression rewrites grammar, not evidence: "implemented" becomes "implementing",
# "merged" becomes "merging". Comparing raw tokens treats every inflection as a new
# fact. Stem candidates are deliberately conservative (a minimum four-character stem,
# suffix stripping only), and they guard nothing on their own — leaked entities,
# numbers, escalation vocabulary, verb drift, and invented outcomes each have their
# own check above that this comparison does not touch.
_INFLECTION_SUFFIXES = (
    "ings", "ing", "edly", "ed", "es", "s",
    "andose", "iendose", "ando", "iendo", "ciones", "cion",
    "ados", "adas", "idos", "idas", "ado", "ada", "ido", "ida",
    "amos", "emos", "imos", "aron", "ieron", "aban", "abas", "aba", "eron",
    "ar", "er", "ir", "io", "os", "as", "o", "a", "e",
)
_MINIMUM_STEM_LENGTH = 4

# Spanish spells one verb's stem differently across its conjugations: reducir gives
# both "reduciendo" and "redujo", buscar gives "buscando" and "busqué". Suffix
# stripping alone leaves those as unrelated stems ("reduc" vs "reduj"), so each pair
# below is normalized at the stem's end. These are orthographic alternations of the
# same verb, never a route to a different word.
_STEM_ALTERNATIONS = (("duj", "duc"), ("qu", "c"), ("gu", "g"), ("z", "c"))


def _word_stems(token: str) -> set[str]:
    """Return one token's conservative stem candidates, including the token itself."""
    stems = {token}
    for suffix in _INFLECTION_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MINIMUM_STEM_LENGTH:
            stems.add(token[: -len(suffix)])
    for stem in tuple(stems):
        for written, alternate in _STEM_ALTERNATIONS:
            if stem.endswith(written):
                stems.add(f"{stem[: -len(written)]}{alternate}")
    return stems


def verify_synthesis_text(
    *,
    text: str,
    selected_facts: list[ResumeFact],
    catalog: list[ResumeFact],
    language: Literal["en", "es"],
    dimension: SynthesisDimension,
    vocabulary_facts: list[ResumeFact] | None = None,
) -> RephraseVerdict:
    """Gate compressed synthesis without requiring one sentence for every fact.

    `selected_facts` is the evidence this text is attributed to; entity leakage, verb
    drift, escalation, and length are judged against it. `vocabulary_facts` is the
    turn's whole authorized selection and bounds word choice, because compression
    draws connective wording across the facts it aggregates. It defaults to
    `selected_facts`, which keeps both boundaries identical for a whole-answer check.
    """
    authorized_facts = vocabulary_facts if vocabulary_facts is not None else selected_facts
    verdict = verify_rephrase(
        text=text,
        selected_facts=selected_facts,
        catalog=catalog,
        language=language,
        require_each_fact_verb=False,
    )
    if not verdict.allowed:
        return verdict

    sentence_count = count_sentences(text)
    word_count = len(text.split())
    if sentence_count > MAX_SYNTHESIS_SENTENCES or word_count > MAX_SYNTHESIS_WORDS:
        return RephraseVerdict(
            allowed=False,
            code="too_long",
            details=[
                f"{sentence_count} sentences / {word_count} words; maximum is "
                f"{MAX_SYNTHESIS_SENTENCES} / {MAX_SYNTHESIS_WORDS}"
            ],
        )

    selected_tokens = {
        token.rstrip(".")
        for fact in authorized_facts
        for token in _fact_vocabulary(fact)
    }
    answer_token_list = [
        token.rstrip(".") for token in normalize_resume_text(text).split()
    ]
    answer_tokens = set(answer_token_list)
    authorized_tokens = (
        selected_tokens
        | _FUNCTION_TOKENS
        | _NAME_TOKENS
        | _authorized_verb_forms(authorized_facts)
    )
    authorized_stems = {
        stem for token in authorized_tokens for stem in _word_stems(token)
    }
    unauthorized = [
        token
        for token in answer_token_list
        if token not in authorized_tokens and not (_word_stems(token) & authorized_stems)
    ]
    if unauthorized:
        return RephraseVerdict(
            allowed=False,
            code="unsupported_vocabulary",
            details=[f"Tokens absent from cited facts: {', '.join(dict.fromkeys(unauthorized))}"],
        )

    expected_markers = answer_tokens & _LANGUAGE_MARKERS[language]
    other_language: Literal["en", "es"] = "es" if language == "en" else "en"
    other_markers = answer_tokens & _LANGUAGE_MARKERS[other_language]
    expected_vocabulary = _language_vocabulary(authorized_facts, language)
    other_vocabulary = _language_vocabulary(authorized_facts, other_language)
    expected_exclusive = answer_tokens & (expected_vocabulary - other_vocabulary)
    other_exclusive = answer_tokens & (other_vocabulary - expected_vocabulary)
    if (
        (not expected_markers and other_markers)
        or len(other_exclusive) > len(expected_exclusive)
    ):
        return RephraseVerdict(
            allowed=False,
            code="wrong_language",
            details=[f"Answer does not follow requested language '{language}'"],
        )
    for concept, forms in _UNSUPPORTED_OUTCOME_CONCEPTS.items():
        if answer_tokens & forms and not selected_tokens & forms:
            return RephraseVerdict(
                allowed=False,
                code="unsupported_outcome",
                details=[f"Outcome concept '{concept}' is absent from selected facts"],
            )

    if dimension == "impact" and not selected_facts:
        return RephraseVerdict(
            allowed=False,
            code="missing_impact_evidence",
            details=["Impact synthesis requires selected explicit-outcome facts"],
        )
    return RephraseVerdict(allowed=True, code="accepted", details=[])


def verify_synthesis_structure(
    *,
    text: str,
    proposition_fact_ids: list[list[str]],
    proposition_texts: list[str],
    dimension: SynthesisDimension,
    detail_requested: bool,
) -> RephraseVerdict:
    """Reject structurally verbose output even when every word is grounded."""
    if dimension == "conclusion" and not detail_requested:
        if len(proposition_texts) > 2 or count_sentences(text) > 2:
            return RephraseVerdict(
                allowed=False,
                code="too_many_examples",
                details=["A concise conclusion may include at most one supporting example"],
            )

    single_fact_sentences = [
        fact_ids
        for proposition, fact_ids in zip(proposition_texts, proposition_fact_ids, strict=True)
        if count_sentences(proposition) == 1 and len(set(fact_ids)) == 1
    ]
    cited_facts = {fact_id for fact_ids in single_fact_sentences for fact_id in fact_ids}
    if len(single_fact_sentences) >= 3 and len(cited_facts) >= 3:
        return RephraseVerdict(
            allowed=False,
            code="fact_dump",
            details=["Synthesis must aggregate facts instead of emitting one sentence per fact"],
        )
    return RephraseVerdict(allowed=True, code="accepted", details=[])
