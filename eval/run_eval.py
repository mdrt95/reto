"""Validate the fixed evaluation dataset before executing an external model run."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.claude import GenerationUnavailableError, create_default_agent_service
from src.config import Settings
from src.models.profile import load_profile


class EvalHistoryItem(BaseModel):
    """A minimal bounded historical turn used by one evaluation scenario."""

    role: str
    content: str


class EvalScenario(BaseModel):
    """A reproducible quality case with explicit expected safety constraints."""

    id: str
    message: str = Field(min_length=1)
    history: list[EvalHistoryItem] = Field(default_factory=list)
    expected_source_ids: list[str]
    tool_required: bool
    inference_permitted: bool
    expected_outcome: Literal["answer", "blocked", "not_found", "clarify"]


def load_scenarios(path: Path) -> list[EvalScenario]:
    """Load and validate the small fixed scenario set before any model expenditure."""
    return [EvalScenario.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def execute_scenarios(scenarios: list[EvalScenario], settings: Settings) -> list[dict[str, object]]:
    """Run the fixed set through the configured core agent and retain only safe metadata."""
    if settings.anthropic_api_key is None:
        raise GenerationUnavailableError("ANTHROPIC_API_KEY is required for an executed evaluation run")
    service = create_default_agent_service(load_profile(settings.profile_path), settings)
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        started_at = perf_counter()
        response = service.respond(
            scenario.message,
            history=[item.model_dump() for item in scenario.history],
        )
        source_ids = set(response.trace.claim_source_ids)
        if scenario.expected_outcome == "blocked":
            outcome_passed = response.trace.guardrail_input == "blocked"
        elif scenario.expected_outcome == "not_found":
            outcome_passed = response.trace.grounding_status == "profile_missing"
        elif scenario.expected_outcome == "clarify":
            outcome_passed = response.trace.grounding_status == "clarification"
        else:
            outcome_passed = (
                response.trace.guardrail_input != "blocked"
                and response.trace.grounding_status not in {"profile_missing", "clarification"}
            )
        source_passed = set(scenario.expected_source_ids).issubset(source_ids)
        tool_passed = not scenario.tool_required or response.trace.tool_name is not None
        results.append(
            {
                "id": scenario.id,
                "outcome_passed": outcome_passed,
                "source_passed": source_passed,
                "tool_passed": tool_passed,
                "grounding_status": response.trace.grounding_status,
                "latency_ms": round((perf_counter() - started_at) * 1_000),
            }
        )
    return results


def main() -> int:
    """Validate scenario structure and print a stable preflight summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("eval/scenarios.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    arguments = parser.parse_args()
    scenarios = load_scenarios(arguments.scenarios)
    if not arguments.execute:
        print(json.dumps({"scenarios_validated": len(scenarios), "status": "ready_for_model_run"}))
        return 0
    settings = Settings()
    results = execute_scenarios(scenarios, settings)
    report = {
        "model": settings.model_name,
        "scenario_count": len(scenarios),
        "results": results,
        "passed": all(
            item["outcome_passed"] and item["source_passed"] and item["tool_passed"]
            for item in results
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "passed": report["passed"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
