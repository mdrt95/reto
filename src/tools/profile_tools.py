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
}


def normalize_resume_text(text: str) -> str:
    """Normalize accents, punctuation, casing, and bounded cross-language aliases."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(character for character in decomposed if not unicodedata.combining(character))
    tokens = re.findall(r"[a-z0-9#+.]+", ascii_text)
    return " ".join(_TOKEN_ALIASES.get(token, token) for token in tokens)


def detect_response_language(text: str) -> Literal["en", "es"]:
    """Choose Spanish only from explicit, bounded lexical evidence."""
    normalized = normalize_resume_text(text)
    spanish_markers = {
        "cual", "que", "como", "con", "construiste", "experiencia", "proyecto",
        "puesto", "tecnologia", "tu", "y",
    }
    return "es" if set(normalized.split()) & spanish_markers else "en"


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
                )
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


def _detect_topic(normalized_query: str, catalog: list[ResumeFact]) -> ResumeTopic | None:
    for topic, phrases in _TOPIC_PHRASES.items():
        if any(normalize_resume_text(phrase) in normalized_query for phrase in phrases):
            return topic
    for fact in catalog:
        if fact.entity and normalize_resume_text(fact.entity) in normalized_query:
            return fact.topic
    return None


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
    query_terms = set(normalized_query.split()) - _STOP_WORDS - topic_words
    scored: list[tuple[int, ResumeFact]] = []
    for fact in candidates:
        haystack = normalize_resume_text(" ".join([fact.text, *fact.keywords, fact.entity or ""]))
        score = sum(1 for term in query_terms if term in haystack)
        scored.append((score, fact))
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
    """Select verified source groups before prose generation for an audience."""
    sources = ["personal", "skills", *[f"experience:{item.id}" for item in profile.experience]]
    if arguments.audience == "technical":
        sources.extend(f"project:{item.id}" for item in profile.projects)
    elif arguments.audience == "executive":
        sources.extend(f"experience:{item.id}" for item in profile.experience if item.current)
    return ProfileSummaryPlan(audience=arguments.audience, source_ids=list(dict.fromkeys(sources)))
