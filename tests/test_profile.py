import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import Settings
from src.main import create_app
from src.models.profile import Profile, ProfileLoadError, load_profile


def test_valid_profile_loads_with_stable_ids() -> None:
    """The approved runtime profile must load and retain its source references."""
    profile = load_profile("data/profile.json")

    assert profile.experience[0].id == "exp-global-payments"
    assert profile.experience[0].highlights[1].id == "hl-security-console"
    assert profile.projects[0].id == "proj-sybil"
    assert profile.projects[0].highlights[0].id == "sybil-hl-rag"
    assert profile.projects[0].highlights[0].technologies == []


def test_invalid_required_profile_data_is_rejected(tmp_path: Path) -> None:
    """A profile without required personal data must never become runtime state."""
    invalid_profile = tmp_path / "invalid-profile.json"
    invalid_profile.write_text(json.dumps({"meta": {"schema_version": "1.0"}}))

    with pytest.raises(ProfileLoadError, match="invalid"):
        load_profile(invalid_profile)


def test_optional_tags_default_to_an_empty_list() -> None:
    """Optional collection fields stay safe to iterate when omitted by source data."""
    profile = Profile.model_validate(
        {
            "meta": {"schema_version": "1.0", "last_updated": "2026-08-31"},
            "personal": {
                "name": "Test User",
                "title": "Engineer",
                "location": "Mexico City",
                "email": "test@example.com",
                "languages": [],
            },
            "skills": {},
            "experience": [],
            "projects": [],
            "education": [],
        }
    )

    assert profile.skills.programming_languages == []


def test_invalid_profile_stops_application_startup(tmp_path: Path) -> None:
    """The service must not report ready when its runtime source is invalid."""
    invalid_profile = tmp_path / "invalid-profile.json"
    invalid_profile.write_text("{}")
    app = create_app(
        Settings(
            environment="test",
            anthropic_api_key="test-key",
            profile_path=invalid_profile,
        )
    )

    with pytest.raises(ProfileLoadError, match="invalid"):
        with TestClient(app):
            pass


def test_production_settings_require_an_api_key() -> None:
    """Production must fail early instead of starting with an unusable provider setup."""
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(environment="production")
