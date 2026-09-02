"""Deterministic, read-only queries over the validated canonical profile."""

import re
import unicodedata
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field

from src.models.profile import Profile


ResumeTopic = Literal[
    "experience",
    "projects",
    "skills",
    "education",
    "languages",
    "summary",
    "career_preferences",
]


class ResumeFact(BaseModel):
    """One profile-derived fact with stable grounding and conversation identifiers."""

    fact_id: str
    source_id: str
    topic: ResumeTopic
    text: str
    entity: str | None = None
    keywords: list[str] = Field(default_factory=list)
    narrative_en: str | None = None
    narrative_es: str | None = None
    field_name: Literal["start_date", "end_date", "current"] | None = None


def fact_display_text(fact: "ResumeFact", language: Literal["en", "es"]) -> str:
    """Return the human-reviewed bilingual narrative when present, else the raw fact text."""
    narrative = fact.narrative_en if language == "en" else fact.narrative_es
    return narrative if narrative else fact.text


class SearchResumeArguments(BaseModel):
    """Validated universal search request over all public resume domains."""

    query: str = Field(min_length=1)
    topic: ResumeTopic | None = None
    source_ids: list[str] = Field(default_factory=list)
    exclude_source_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=20)


class ResumeSearchResult(BaseModel):
    """Selected facts plus routing metadata for deterministic recovery."""

    query: str
    language: Literal["en", "es"]
    topic: ResumeTopic
    matches: list[ResumeFact] = Field(default_factory=list)
    profile_missing: bool = False
    unmatched_terms: list[str] = Field(default_factory=list)
    """Query terms that matched no candidate fact. A retrieval diagnostic only.

    These are raw terms, not entities: a pronoun, a verb, or a punctuation artifact
    lands here as readily as a company name. Never render them in an answer.
    """


class ProjectMatch(BaseModel):
    """A sourceable project highlight returned by deterministic search."""

    source_id: str
    project_name: str
    summary: str


class ProjectSearchResult(BaseModel):
    """Results from a project technology or keyword lookup."""

    matches: list[ProjectMatch] = Field(default_factory=list)


class SearchProjectsArguments(BaseModel):
    """Validated input for deterministic project search."""

    query: str = Field(min_length=1)


class ExperienceMatch(BaseModel):
    """A sourceable experience highlight returned by a typed filter."""

    experience_id: str
    highlight_id: str
    source_id: str
    summary: str


class ExperienceFilterResult(BaseModel):
    """Results from an allowlisted experience filter."""

    matches: list[ExperienceMatch] = Field(default_factory=list)


class FilterExperienceArguments(BaseModel):
    """Validated, allowlisted input for experience filtering."""

    filter_by: Literal["technology", "tag", "role"]
    value: str = Field(min_length=1)


class ProfileQueryResult(BaseModel):
    """A public, typed profile projection without unrestricted contact data."""

    field: str
    value: list[str]
    source_ids: list[str] = Field(default_factory=list)


class QueryProfileArguments(BaseModel):
    """Validated public projection selector."""

    field: Literal["skills", "languages", "education", "current_role", "companies"]


class ProfileSummaryPlan(BaseModel):
    """Verified source selection for a model-generated audience summary."""

    audience: str
    source_ids: list[str]
    fact_ids: list[str] = Field(default_factory=list)


class SummarizeProfileArguments(BaseModel):
    """Validated audience selector for a source-backed summary."""

    audience: Literal["technical", "recruiter", "executive"]


_TOPIC_PHRASES: dict[ResumeTopic, tuple[str, ...]] = {
    "career_preferences": (
        "looking for", "desired role", "career preference", "tipo de puesto",
        "puesto buscas", "puesto estas buscando", "buscando trabajo", "preferencias profesionales",
    ),
    "projects": (
        "project", "proyecto", "built", "build", "construido", "construiste",
        "worked in", "worked on", "has hecho",
    ),
    "experience": (
        "experience", "experiencia", "work history", "employment", "empleo",
        "trabajo", "worked", "career", "carrera",
    ),
    "skills": (
        "skill", "technology", "technologies", "stack", "habilidad", "tecnologia",
        "tecnologias", "dominas", "herramientas",
    ),
    "education": ("education", "degree", "university", "educacion", "estudios", "universidad"),
    "languages": ("languages", "language", "idiomas", "idioma", "hablas"),
    "summary": (
        "tell me about yourself", "about marco", "professional summary", "resume",
        "profile", "hablame de ti", "sobre ti", "quien eres", "resumen profesional",
    ),
}

_TOKEN_ALIASES = {
    "recuperacion": "retrieval",
    "semantica": "semantic",
    "semantico": "semantic",
    "inteligencia": "ai",
    "artificial": "ai",
    "seguridad": "security",
    "proyectos": "project",
    "proyecto": "project",
    "projects": "project",
    "tecnologias": "technology",
    "tecnologia": "technology",
}

_STOP_WORDS = {
    "a", "about", "cual", "cuales", "de", "del", "el", "en", "es", "has", "have",
    "i", "in", "is", "la", "las", "lo", "los", "marco", "me", "mi", "que", "tell",
    "the", "tu", "tus", "what", "which", "your", "you",
    "y",
    "at", "with", "for", "on", "from", "to", "of", "did", "do", "does", "was",
    "were", "are", "be", "been", "any", "s", "con", "para", "por", "sobre",
    "desde", "hasta", "ha", "han", "tiene", "fue", "son", "un", "una", "al",
    # Pronouns and determiners. They carry no retrieval signal, and left in they
    # both empty a stocked topic and get rendered as if they were missing entities.
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "it", "its", "we", "us", "our", "this", "that", "these", "those",
    "su", "sus", "ella", "ellos", "ellas", "nos", "nuestro", "nuestra", "ese",
    "esa", "eso", "esos", "esas", "este", "esta", "esto", "estos", "estas",
    # Discourse verbs and reference nouns: they talk about the conversation, not
    # about the profile.
    "part", "parte", "where", "donde", "when", "cuando", "say", "says", "said",
    "dice", "dijo", "dijiste", "mention", "mentions", "mentioned", "menciona",
    "mencionaste", "more", "mas",
    # Request verbs naming the corpus or the operation, never a search term.
    "resume", "resumen", "resumir", "resume.", "summarize", "summary", "dime",
    "cuentame", "acerca", "give", "show", "please", "por favor",
}


def normalize_resume_text(text: str) -> str:
    """Normalize accents, punctuation, casing, and bounded cross-language aliases."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(character for character in decomposed if not unicodedata.combining(character))
    tokens = re.findall(r"[a-z0-9#+.]+", ascii_text)
    # A dot inside a token is part of a product name (asp.net, node.js); a dot at
    # either edge is sentence punctuation, and leaving it there makes "marco." a
    # different token from "marco".
    trimmed = (token.strip(".") for token in tokens)
    return " ".join(_TOKEN_ALIASES.get(token, token) for token in trimmed if token)


_SPANISH_ONLY_CHARACTERS = frozenset("ñáéíóúü¿¡")
"""Orthography no English sentence contains. Contributing evidence, never decisive:
an unaccented sentence must reach the same verdict as its accented form."""

_SPANISH_FUNCTION_WORDS = frozenset({
    "que", "qué", "cual", "cuales", "como", "donde", "cuando", "quien", "quienes",
    "cuanto", "cuanta", "cuantos", "cuantas", "desde", "hasta", "para", "por",
    "porque", "pero", "tambien", "ademas", "entre", "hacia", "segun", "sin",
    "sobre", "tras", "durante", "cada", "todo", "toda", "todos", "todas", "otro",
    "otra", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aqui", "ahi", "alli", "alla", "de", "del", "la", "el", "los", "las",
    "un", "una", "unos", "unas", "su", "sus", "tu", "tus", "mi", "mis", "y", "o",
    "es", "son", "era", "fue", "fueron", "ser", "estar", "tiene", "tienen",
    "hay", "muy", "mas", "menos", "bien", "en", "con", "al", "lo", "le", "les",
    "se", "ya", "tiempo", "trabaja", "habla", "quiero", "saber", "dime",
    "platicame", "cuentame", "hablame", "explicame", "muestrame", "dame",
})
"""Closed-class Spanish words. A contributor to the score, not the classifier."""

_SPANISH_STEMS = (
    "proyecto", "experiencia", "tecnolog", "habilidad", "idioma", "estudi",
    "trabaj", "logro", "empresa", "puesto", "carrera", "ingenier", "construi",
    "resumen", "certificacion", "educacion",
)
"""Open-class stems, matched by prefix so plurals and inflections are covered."""

_SPANISH_SUFFIXES = (
    "cion", "ciones", "dad", "dades", "mente", "ando", "iendo", "aron", "aste",
    "iste", "amos", "emos", "aban", "aria", "eria",
)
"""Morphology carrying Spanish regardless of vocabulary. Applied only to longer
tokens, where English collisions ("waste", "baron", "dad") cannot reach."""

_ENGLISH_FUNCTION_WORDS = frozenset({
    "what", "which", "where", "when", "who", "whom", "how", "why", "is", "are",
    "was", "were", "do", "does", "did", "have", "had", "the", "an", "of", "for",
    "in", "on", "at", "to", "from", "with", "about", "your", "yours", "you",
    "yourself", "his", "her", "their", "there", "that", "this", "these", "those",
    "tell", "show", "been", "being", "can", "could", "would", "should", "and",
    "but", "long", "much", "many", "any", "all", "give", "list", "summarize",
})
"""Deliberately excludes tokens both languages share ("a", "no", "me", "has"),
which would otherwise score English for a Spanish sentence."""


def _language_tokens(text: str) -> list[str]:
    """Split on non-alphanumerics after accent folding.

    Unlike `normalize_resume_text` this keeps no punctuation inside a token, so a
    sentence-final marker ("proyecto.") is still that marker, and applies no cross
    language aliases, which would erase the very evidence being weighed.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.findall(r"[a-z0-9]+", folded)


def detect_response_language(text: str) -> Literal["en", "es"]:
    """Weigh orthographic, morphological, and lexical evidence for the reply language.

    A closed marker list cannot be the classifier: it omits whatever nobody thought
    to type, and every omission answers a Spanish question in English. Evidence is
    scored instead, and English wins ties so an unrecognized sentence keeps the
    previous default.
    """
    tokens = _language_tokens(text)
    spanish = sum(1 for token in tokens if token in _SPANISH_FUNCTION_WORDS)
    spanish += sum(1 for token in tokens if token.startswith(_SPANISH_STEMS))
    spanish += sum(
        1
        for token in tokens
        if len(token) >= 6 and token.endswith(_SPANISH_SUFFIXES)
    )
    if any(character in _SPANISH_ONLY_CHARACTERS for character in text.casefold()):
        spanish += 1
    english = sum(1 for token in tokens if token in _ENGLISH_FUNCTION_WORDS)
    return "es" if spanish > english else "en"


def _source_related(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


def _stable_value_key(value: str) -> str:
    """Create a readable, reorder-independent key from canonical profile content."""
    normalized = normalize_resume_text(value)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:36] or "value"
    digest = sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def build_resume_fact_catalog(profile: Profile) -> list[ResumeFact]:
    """Build every searchable fact from model data; no parallel truth is stored."""
    facts: list[ResumeFact] = [
        ResumeFact(
            fact_id="fact:personal:title",
            source_id="personal",
            topic="summary",
            text=f"{profile.personal.name} — {profile.personal.title}",
            entity=profile.personal.name,
            keywords=[profile.personal.title, profile.personal.location],
        )
    ]
    if profile.professional_summary:
        facts.append(
            ResumeFact(
                fact_id="fact:professional_summary",
                source_id="professional_summary",
                topic="summary",
                text=profile.professional_summary,
                entity=profile.personal.name,
            )
        )
    for category, values in profile.skills.model_dump().items():
        for value in values:
            facts.append(
                ResumeFact(
                    fact_id=f"fact:skills:{category}:{_stable_value_key(value)}",
                    source_id="skills",
                    topic="skills",
                    text=value,
                    entity=value,
                    keywords=[category.replace("_", " ")],
                )
            )
    for language in profile.personal.languages:
        facts.append(
            ResumeFact(
                fact_id=f"fact:personal:language:{_stable_value_key(language.language)}",
                source_id="personal",
                topic="languages",
                text=f"{language.language} ({language.level})",
                entity=language.language,
            )
        )
    for experience in profile.experience:
        source_id = f"experience:{experience.id}"
        facts.append(
            ResumeFact(
                fact_id=f"fact:{source_id}",
                source_id=source_id,
                topic="experience",
                text=f"{experience.role} at {experience.company}. {experience.team_context}",
                entity=experience.company,
                keywords=[experience.role, experience.company, "current" if experience.current else "past"],
                narrative_en=experience.narrative.en if experience.narrative else None,
                narrative_es=experience.narrative.es if experience.narrative else None,
            )
        )
        facts.append(
            ResumeFact(
                fact_id=f"fact:summary:{experience.id}",
                source_id=source_id,
                topic="summary",
                text=f"{experience.role} at {experience.company}",
                entity=experience.company,
                keywords=[experience.role, "experience"],
            )
        )
        for highlight in experience.highlights:
            highlight_source = f"{source_id}.highlight:{highlight.id}"
            facts.append(
                ResumeFact(
                    fact_id=f"fact:{highlight_source}",
                    source_id=highlight_source,
                    topic="experience",
                    text=f"{highlight.summary} {highlight.detail}".strip(),
                    entity=experience.company,
                    keywords=[*highlight.technologies, *highlight.tags, experience.role],
                    narrative_en=highlight.narrative.en if highlight.narrative else None,
                    narrative_es=highlight.narrative.es if highlight.narrative else None,
                )
            )
        # Field projections follow the record and its highlights so existing
        # source-ordered narrative selection remains stable.
        facts.extend(
            [
                ResumeFact(
                    fact_id=f"fact:{source_id}:start_date",
                    source_id=source_id,
                    topic="experience",
                    text=experience.start_date,
                    entity=experience.company,
                    keywords=[experience.company, "start date", "fecha de inicio"],
                    field_name="start_date",
                ),
                ResumeFact(
                    fact_id=f"fact:{source_id}:end_date",
                    source_id=source_id,
                    topic="experience",
                    text=experience.end_date or "current role",
                    entity=experience.company,
                    keywords=[experience.company, "end date", "fecha de fin"],
                    field_name="end_date",
                ),
                ResumeFact(
                    fact_id=f"fact:{source_id}:current",
                    source_id=source_id,
                    topic="experience",
                    text="current" if experience.current else "not current",
                    entity=experience.company,
                    keywords=[experience.company, "current", "actual"],
                    field_name="current",
                ),
            ]
        )
    for project in profile.projects:
        source_id = f"project:{project.id}"
        facts.append(
            ResumeFact(
                fact_id=f"fact:{source_id}",
                source_id=source_id,
                topic="projects",
                text=f"{project.name}: {project.subtitle}",
                entity=project.name,
                keywords=[*project.technologies, project.status],
                narrative_en=project.narrative.en if project.narrative else None,
                narrative_es=project.narrative.es if project.narrative else None,
            )
        )
        for highlight in project.highlights:
            highlight_source = f"{source_id}.highlight:{highlight.id}"
            facts.append(
                ResumeFact(
                    fact_id=f"fact:{highlight_source}",
                    source_id=highlight_source,
                    topic="projects",
                    text=f"{highlight.summary} {highlight.detail}".strip(),
                    entity=project.name,
                    keywords=[*project.technologies, *highlight.technologies, *highlight.tags],
                    narrative_en=highlight.narrative.en if highlight.narrative else None,
                    narrative_es=highlight.narrative.es if highlight.narrative else None,
                )
            )
    for education in profile.education:
        source_id = f"education:{education.id}"
        facts.append(
            ResumeFact(
                fact_id=f"fact:{source_id}",
                source_id=source_id,
                topic="education",
                text=f"{education.degree} — {education.institution} ({education.start_year}–{education.end_year})",
                entity=education.institution,
                narrative_en=education.narrative.en if education.narrative else None,
                narrative_es=education.narrative.es if education.narrative else None,
            )
        )
    if profile.career_preferences:
        preference_values = {
            "desired_roles": profile.career_preferences.desired_roles,
            "seniority": [profile.career_preferences.seniority] if profile.career_preferences.seniority else [],
            "locations": profile.career_preferences.locations,
            "work_arrangements": profile.career_preferences.work_arrangements,
            "notes": profile.career_preferences.notes,
        }
        for field_name, values in preference_values.items():
            for value in values:
                facts.append(
                    ResumeFact(
                        fact_id=f"fact:career_preferences:{field_name}:{_stable_value_key(value)}",
                        source_id="career_preferences",
                        topic="career_preferences",
                        text=value,
                        entity=value,
                        keywords=[field_name.replace("_", " ")],
                    )
                )
    return facts


_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9\-\.]+")
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9][A-Za-zÁÉÍÓÚÑáéíóúñ0-9\-\.']*")
# Split only on sentence punctuation; a dot inside a token (Node.js, ASP.NET) is not a boundary.
_SENTENCE_SPLIT_RE = re.compile(r"[?!¿¡]+|\.(?=\s|$)")
_TITLE_CASE_RATIO = 0.6


def _profile_known_tokens(profile: Profile) -> set[str]:
    """Build the normalized-token vocabulary supported anywhere in the profile."""
    catalog = build_resume_fact_catalog(profile)
    parts: list[str] = []
    for fact in catalog:
        parts.append(fact.text)
        parts.extend(fact.keywords)
        if fact.entity:
            parts.append(fact.entity)
        if fact.narrative_en:
            parts.append(fact.narrative_en)
        if fact.narrative_es:
            parts.append(fact.narrative_es)
    tokens = set(normalize_resume_text(" ".join(parts)).split())
    # Dotted product names are also known by their pieces (asp.net -> asp, net).
    tokens.update(piece for token in list(tokens) for piece in token.split(".") if piece)
    return tokens


def _is_title_case(sentence: str) -> bool:
    """Detect prose where capitalization is stylistic, not a proper-noun signal.

    The opening word is capitalized by orthography in every sentence, so counting it
    makes a short question look title-cased purely because it is short: `Did Marco
    work at Google?` reached the ratio and skipped entity detection entirely, while
    the same entity in a longer phrasing was caught. Only the words after the
    opening one carry a stylistic-capitalization signal.
    """
    words = _WORD_RE.findall(sentence)
    if len(words) < 3:
        return False
    body = words[1:]
    capitalized = sum(1 for word in body if word[0].isupper())
    return capitalized / len(body) >= _TITLE_CASE_RATIO


def find_unknown_entities(profile: Profile, message: str) -> list[str]:
    """Find capitalized, non-sentence-initial tokens absent from every profile fact.

    A named entity the profile never mentions must produce a not-found answer rather
    than an answer assembled from unrelated verified facts. Sentence-initial words and
    title-case prose are skipped because their capitalization carries no entity signal.
    """
    name_tokens = {normalize_resume_text(part) for part in profile.personal.name.split()}
    known_tokens = _profile_known_tokens(profile)
    unknown: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(message):
        if _is_title_case(sentence):
            continue
        first_word = _WORD_RE.search(sentence)
        sentence_start = first_word.start() if first_word else -1
        for match in _CAPITALIZED_TOKEN_RE.finditer(sentence):
            if match.start() == sentence_start:
                continue
            raw_token = match.group(0).rstrip(".-")
            if raw_token.endswith("'s") or raw_token.endswith("\u2019s"):
                raw_token = raw_token[:-2]
            normalized_token = normalize_resume_text(raw_token)
            if not normalized_token or normalized_token in name_tokens or normalized_token in _STOP_WORDS:
                continue
            singular = normalized_token[:-1] if normalized_token.endswith("s") and len(normalized_token) > 1 else normalized_token
            if normalized_token in known_tokens or singular in known_tokens:
                continue
            if normalized_token in seen:
                continue
            seen.add(normalized_token)
            unknown.append(raw_token)
    return unknown


def _detect_topic(normalized_query: str, catalog: list[ResumeFact]) -> ResumeTopic | None:
    for topic, phrases in _TOPIC_PHRASES.items():
        if any(normalize_resume_text(phrase) in normalized_query for phrase in phrases):
            return topic
    for fact in catalog:
        if fact.entity and normalize_resume_text(fact.entity) in normalized_query:
            return fact.topic
    return None


def detect_resume_topic(profile: Profile, message: str) -> ResumeTopic | None:
    """Resolve the topic a message anchors to, or None when no anchor is evidenced.

    `search_resume` deliberately widens an unresolved anchor to the broad `summary`
    topic so it always returns something. A caller deciding whether recovery is even
    permitted must see the unresolved case, because substituting summary facts for an
    unanchored question is how unrelated facts reach an answer.
    """
    return _detect_topic(normalize_resume_text(message), build_resume_fact_catalog(profile))


def search_resume(profile: Profile, arguments: SearchResumeArguments) -> ResumeSearchResult:
    """Search every public resume domain using a derived, normalized fact catalog."""
    catalog = build_resume_fact_catalog(profile)
    normalized_query = normalize_resume_text(arguments.query)
    topic = arguments.topic or _detect_topic(normalized_query, catalog)
    if topic is None:
        topic = "summary"
    candidates = [fact for fact in catalog if fact.topic == topic]
    if arguments.source_ids:
        candidates = [
            fact
            for fact in candidates
            if any(_source_related(fact.source_id, selected) for selected in arguments.source_ids)
        ]
    if arguments.exclude_source_ids:
        candidates = [
            fact
            for fact in candidates
            if fact.source_id not in arguments.exclude_source_ids
        ]
    topic_words = {
        word
        for phrase in _TOPIC_PHRASES[topic]
        for word in normalize_resume_text(phrase).split()
    }
    query_term_list = [
        term
        for term in dict.fromkeys(normalized_query.split())
        if term not in _STOP_WORDS and term not in topic_words
    ]
    query_terms = set(query_term_list)
    term_matched = {term: False for term in query_term_list}
    scored: list[tuple[int, ResumeFact]] = []
    for fact in candidates:
        haystack = normalize_resume_text(" ".join([fact.text, *fact.keywords, fact.entity or ""]))
        score = 0
        for term in query_terms:
            if term in haystack:
                score += 1
                term_matched[term] = True
        scored.append((score, fact))
    unmatched_terms = [term for term in query_term_list if not term_matched[term]]
    if query_terms and any(score for score, _ in scored):
        candidates = [fact for score, fact in scored if score > 0]
    elif query_terms:
        candidates = []
    return ResumeSearchResult(
        query=arguments.query,
        language=detect_response_language(arguments.query),
        topic=topic,
        matches=candidates[: arguments.limit],
        profile_missing=not candidates,
        unmatched_terms=unmatched_terms,
    )


def _contains_query(query: str, values: list[str]) -> bool:
    """Match normalized user text against normalized structured profile values."""
    normalized_query = query.casefold().strip()
    return bool(normalized_query) and normalized_query in " ".join(values).casefold()


def search_projects(profile: Profile, arguments: SearchProjectsArguments) -> ProjectSearchResult:
    """Find project highlights containing a requested technology or keyword."""
    matches: list[ProjectMatch] = []
    for project in profile.projects:
        project_matches = _contains_query(
            arguments.query,
            [project.name, project.subtitle, *project.technologies],
        )
        matching_highlight_found = False
        for highlight in project.highlights:
            searchable = [
                project.name,
                project.subtitle,
                highlight.summary,
                highlight.detail,
                *highlight.technologies,
                *highlight.tags,
            ]
            if _contains_query(arguments.query, searchable):
                matching_highlight_found = True
                matches.append(
                    ProjectMatch(
                        source_id=f"project:{project.id}.highlight:{highlight.id}",
                        project_name=project.name,
                        summary=highlight.summary,
                    )
                )
        if project_matches and not matching_highlight_found:
            matches.append(
                ProjectMatch(
                    source_id=f"project:{project.id}",
                    project_name=project.name,
                    summary=project.subtitle,
                )
            )
    return ProjectSearchResult(matches=matches)


def filter_experience(profile: Profile, arguments: FilterExperienceArguments) -> ExperienceFilterResult:
    """Filter experience by an explicit, allowlisted structured field."""
    matches: list[ExperienceMatch] = []
    for experience in profile.experience:
        for highlight in experience.highlights:
            values = {
                "technology": highlight.technologies,
                "tag": highlight.tags,
                "role": [experience.role],
            }[arguments.filter_by]
            if _contains_query(arguments.value, values):
                matches.append(
                    ExperienceMatch(
                        experience_id=experience.id,
                        highlight_id=highlight.id,
                        source_id=f"experience:{experience.id}.highlight:{highlight.id}",
                        summary=highlight.summary,
                    )
                )
    return ExperienceFilterResult(matches=matches)


def query_profile(profile: Profile, arguments: QueryProfileArguments) -> ProfileQueryResult:
    """Return an allowlisted public projection of profile data."""
    companies = list(dict.fromkeys(item.company for item in profile.experience))
    values = {
        "skills": [
            *profile.skills.programming_languages,
            *profile.skills.ai_llm,
            *profile.skills.ai_stack,
            *profile.skills.backend_apis,
            *profile.skills.devops_engineering,
        ],
        "languages": [f"{item.language} ({item.level})" for item in profile.personal.languages],
        "education": [f"{item.degree} — {item.institution}" for item in profile.education],
        "current_role": [
            f"{item.role} at {item.company}"
            for item in profile.experience
            if item.current
        ],
        "companies": companies,
    }
    sources = {
        "skills": ["skills"],
        "languages": ["personal"],
        "education": [f"education:{item.id}" for item in profile.education],
        "current_role": [
            f"experience:{item.id}" for item in profile.experience if item.current
        ],
        "companies": [f"experience:{item.id}" for item in profile.experience],
    }
    return ProfileQueryResult(
        field=arguments.field,
        value=values[arguments.field],
        source_ids=list(dict.fromkeys(sources[arguments.field])),
    )


def summarize_profile(profile: Profile, arguments: SummarizeProfileArguments) -> ProfileSummaryPlan:
    """Select verified source groups before prose generation for an audience.

    `fact_ids` (D-031) is a deterministic, ordered, <= 8-entry fact selection so the
    orchestrator can skip the generator's fact-selection call entirely for summaries,
    the same way it already does for small tool-narrowed fact sets.
    """
    sources = ["personal", "skills", *[f"experience:{item.id}" for item in profile.experience]]
    if arguments.audience == "technical":
        sources.extend(f"project:{item.id}" for item in profile.projects)
    elif arguments.audience == "executive":
        sources.extend(f"experience:{item.id}" for item in profile.experience if item.current)

    fact_ids: list[str] = [f"fact:experience:{item.id}" for item in profile.experience]
    if arguments.audience == "recruiter":
        for experience in profile.experience:
            fact_ids.extend(
                f"fact:experience:{experience.id}.highlight:{highlight.id}"
                for highlight in experience.highlights[:3]
            )
        fact_ids.extend(f"fact:education:{item.id}" for item in profile.education)
    elif arguments.audience == "technical":
        for experience in profile.experience:
            fact_ids.extend(
                f"fact:experience:{experience.id}.highlight:{highlight.id}"
                for highlight in experience.highlights[:2]
            )
        for project in profile.projects:
            fact_ids.append(f"fact:project:{project.id}")
            fact_ids.extend(
                f"fact:project:{project.id}.highlight:{highlight.id}"
                for highlight in project.highlights[:2]
            )
    elif arguments.audience == "executive":
        for experience in profile.experience:
            if not experience.current:
                continue
            fact_ids.append(f"fact:experience:{experience.id}")
            fact_ids.extend(
                f"fact:experience:{experience.id}.highlight:{highlight.id}"
                for highlight in experience.highlights[:2]
            )
        fact_ids.extend(f"fact:education:{item.id}" for item in profile.education)

    return ProfileSummaryPlan(
        audience=arguments.audience,
        source_ids=list(dict.fromkeys(sources)),
        fact_ids=list(dict.fromkeys(fact_ids))[:8],
    )
