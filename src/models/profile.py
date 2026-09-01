"""Pydantic models and loader for the approved runtime profile."""

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ProfileLoadError(RuntimeError):
    """Raised when the approved runtime profile cannot be read or validated."""


class ProfileModel(BaseModel):
    """Shared strict validation behavior for all profile boundaries."""

    model_config = ConfigDict(extra="forbid")


class Metadata(ProfileModel):
    """Schema metadata for a versioned runtime profile."""

    schema_version: str
    last_updated: date


class Language(ProfileModel):
    """A language and the stated professional proficiency."""

    language: str
    level: str


class PersonalData(ProfileModel):
    """Private source personal data; routes must choose safe public fields explicitly."""

    name: str
    title: str
    location: str
    email: str
    phone: str | None = None
    languages: list[Language]


class Skills(ProfileModel):
    """Categorized technical skills with safe collection defaults."""

    programming_languages: list[str] = Field(default_factory=list)
    ai_llm: list[str] = Field(default_factory=list)
    ai_stack: list[str] = Field(default_factory=list)
    backend_apis: list[str] = Field(default_factory=list)
    devops_engineering: list[str] = Field(default_factory=list)


class Narrative(ProfileModel):
    """Human-reviewed bilingual prose restating a fact's existing text only."""

    en: str = Field(min_length=1)
    es: str = Field(min_length=1)


class ExperienceHighlight(ProfileModel):
    """A stable, sourceable outcome from a professional experience."""

    id: str
    summary: str
    detail: str
    technologies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    narrative: Narrative | None = None


class Experience(ProfileModel):
    """Employment record and its stable supporting highlights."""

    id: str
    role: str
    company: str
    start_date: str
    end_date: str | None
    current: bool
    team_context: str
    highlights: list[ExperienceHighlight]
    narrative: Narrative | None = None


class ProjectHighlight(ProfileModel):
    """A stable, sourceable outcome from a project."""

    id: str
    summary: str
    detail: str
    technologies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    narrative: Narrative | None = None


class Project(ProfileModel):
    """Portfolio project and its stable supporting highlights."""

    id: str
    name: str
    subtitle: str
    status: str
    technologies: list[str] = Field(default_factory=list)
    highlights: list[ProjectHighlight]
    narrative: Narrative | None = None


class Education(ProfileModel):
    """Formal education record from the approved profile."""

    id: str
    degree: str
    institution: str
    start_year: int
    end_year: int
    narrative: Narrative | None = None


class CareerPreferences(ProfileModel):
    """Optional, explicitly stated career preferences; absence is meaningful."""

    desired_roles: list[str] = Field(default_factory=list)
    seniority: str | None = None
    locations: list[str] = Field(default_factory=list)
    work_arrangements: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Profile(ProfileModel):
    """Complete validated professional profile used as runtime truth."""

    meta: Metadata
    personal: PersonalData
    skills: Skills
    experience: list[Experience]
    projects: list[Project]
    education: list[Education]
    professional_summary: str | None = None
    career_preferences: CareerPreferences | None = None


def load_profile(profile_path: str | Path) -> Profile:
    """Load the approved JSON source once so invalid profile data stops startup."""
    path = Path(profile_path)
    try:
        raw_profile = json.loads(path.read_text(encoding="utf-8"))
        return Profile.model_validate(raw_profile)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ProfileLoadError(f"Profile at {path} is missing or invalid: {error}") from error
