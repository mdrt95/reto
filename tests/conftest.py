from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a started application with explicit test-only settings."""
    settings = Settings(
        environment="test",
        anthropic_api_key="test-key",
        profile_path="data/profile.json",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
