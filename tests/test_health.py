def test_health_returns_ready_version_without_configuration(client) -> None:
    """The health endpoint should expose only non-sensitive readiness metadata."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"version": "0.1.0", "ready": True}
    assert "test-key" not in response.text
    assert "Marco Reyes" not in response.text
