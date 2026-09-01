"""Readiness endpoint with no profile or configuration disclosure."""

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    """Safe public readiness metadata."""

    version: str
    ready: bool


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Report whether startup completed without exposing internal configuration."""
    return HealthResponse(version=request.app.version, ready=bool(request.app.state.ready))
