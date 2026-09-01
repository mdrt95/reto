"""Deterministic, read-only queries over the small structured CV corpus."""

from typing import Literal

from pydantic import BaseModel, Field

from src.models.profile import Profile


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


class QueryProfileArguments(BaseModel):
    """Validated public projection selector."""

    field: Literal["skills", "languages", "education", "current_role"]


class ProfileSummaryPlan(BaseModel):
    """Verified source selection for a model-generated audience summary."""

    audience: str
    source_ids: list[str]


class SummarizeProfileArguments(BaseModel):
    """Validated audience selector for a source-backed summary."""

    audience: Literal["technical", "recruiter", "executive"]


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
    }
    return ProfileQueryResult(field=arguments.field, value=values[arguments.field])


def summarize_profile(profile: Profile, arguments: SummarizeProfileArguments) -> ProfileSummaryPlan:
    """Select verified source groups before prose generation for an audience."""
    sources = ["personal", "skills", *[f"experience:{item.id}" for item in profile.experience]]
    if arguments.audience == "technical":
        sources.extend(f"project:{item.id}" for item in profile.projects)
    elif arguments.audience == "executive":
        sources.extend(f"experience:{item.id}" for item in profile.experience if item.current)
    return ProfileSummaryPlan(audience=arguments.audience, source_ids=list(dict.fromkeys(sources)))
