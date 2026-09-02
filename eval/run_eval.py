"""Enforce the answer contract of specifications 01 and 03 against a fixed scenario matrix.

Automated evaluation covers the guarantees a reader cannot see: which canonical facts a
turn selected, which provider stages it was allowed to call, whether an answer was
rendered deterministically or transformed, and how a deliberate provider outage behaves.
Conversational quality is not asserted here — the fixed bilingual UI script in
`specs/03-quality-operations-and-deployment.md` remains the second, required gate.
"""

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from anthropic import Anthropic

from src.agent.claude import (
    ClaudeIntentClassifier,
    ClaudeRephraser,
    ClaudeResponseGenerator,
    GenerationUnavailableError,
)
from src.agent.contracts import Intent, IntentDecision
from src.agent.orchestrator import AgentService
from src.config import Settings
from src.models.profile import load_profile

# A deterministic rendering never delivers model-authored prose: the text is
# reconstructed from canonical profile facts. `transformed` is the only mode whose
# wording the provider chose, so it is the only mode an inference can reach.
DETERMINISTIC_RENDERING_MODES = frozenset(
    {
        "canonical",
        "canonical_fallback",
        "canonical_not_found",
        "clarification",
        "informativeness_fallback",
    }
)

# Coarse classifier fields that must lose to explicit evidence in the current message.
CLASSIFIER_DOUBLES: dict[str, IntentDecision] = {
    "companies": IntentDecision(
        intent=Intent.DIRECT_QUESTION, confidence=0.99, profile_field="companies"
    ),
    "summary_request": IntentDecision(
        intent=Intent.SUMMARY_REQUEST, confidence=0.99, audience="recruiter"
    ),
    "skills": IntentDecision(
        intent=Intent.DIRECT_QUESTION, confidence=0.99, profile_field="skills"
    ),
}


class EvalHistoryItem(BaseModel):
    """A minimal bounded historical turn used by one evaluation scenario."""

    role: str
    content: str


class EvalScenario(BaseModel):
    """One reproducible case and the complete contract its answer must satisfy."""

    id: str
    message: str = Field(min_length=1)
    history: list[EvalHistoryItem] = Field(default_factory=list)

    # Deliberate doubles. A classifier double proves the current message outranks a
    # coarse field; an outage proves the deterministic floor holds without a provider.
    classifier_double: Literal["companies", "summary_request", "skills"] | None = None
    provider_outage: Literal["classifier", "generator", "rephraser"] | None = None

    expected_outcome: Literal["answer", "blocked", "not_found", "clarify"]
    expected_status: int = 200
    expected_language: Literal["en", "es"] | None = None
    expected_answer_mode: Literal["direct", "synthesis"] | None = None
    expected_rendering_mode: str | None = None
    expected_topic: str | None = None
    expected_tool: str | None = None
    tool_required: bool = False

    inference_permitted: bool = False
    max_sentences: int | None = None
    max_words: int | None = None
    max_generation_calls: int | None = None

    # Exact sets: an unexpected extra fact is a failure, not a tolerated superset.
    expected_fact_ids: list[str] = Field(default_factory=list)
    forbidden_fact_ids: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    forbidden_source_ids: list[str] = Field(default_factory=list)
    required_tokens: list[str] = Field(default_factory=list)
    forbidden_tokens: list[str] = Field(default_factory=list)


_TOKEN_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def token_present(token: str, text: str) -> bool:
    """Check a required/forbidden eval token as a whole word or phrase, case-insensitively.

    A plain substring check false-positives on a token that is a prefix of an unrelated
    longer word (e.g. forbidden "led" inside "scheduled").
    """
    pattern = _TOKEN_PATTERN_CACHE.get(token)
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        _TOKEN_PATTERN_CACHE[token] = pattern
    return pattern.search(text) is not None


def load_scenarios(path: Path) -> list[EvalScenario]:
    """Load and validate the fixed scenario matrix before any model expenditure."""
    return [
        EvalScenario.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


class _FixedClassifier:
    """Return one deliberately incompatible decision for every message."""

    def __init__(self, decision: IntentDecision) -> None:
        self._decision = decision

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        return self._decision


class _OutageStage:
    """Fail one named provider stage deterministically, with no network access."""

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        raise GenerationUnavailableError("classifier unavailable")

    def generate(self, **_: object) -> Any:
        raise GenerationUnavailableError("generator unavailable")

    def rephrase(self, **_: object) -> Any:
        raise GenerationUnavailableError("rephraser unavailable")


class _CountingStage:
    """Wrap a provider stage so the runner can assert how often it was called."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls = 0

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        self.calls += 1
        return self._inner.classify(message, history)  # type: ignore[attr-defined]

    def generate(self, **kwargs: object) -> Any:
        self.calls += 1
        return self._inner.generate(**kwargs)  # type: ignore[attr-defined]

    def rephrase(self, **kwargs: object) -> Any:
        self.calls += 1
        return self._inner.rephrase(**kwargs)  # type: ignore[attr-defined]


def build_service(scenario: EvalScenario, settings: Settings) -> tuple[AgentService, dict[str, _CountingStage]]:
    """Assemble the service this scenario needs, substituting its declared doubles."""
    profile = load_profile(settings.profile_path)
    client = Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=settings.model_timeout_seconds,
    )
    classifier: object = ClaudeIntentClassifier(client=client, settings=settings)
    generator: object = ClaudeResponseGenerator(client=client, settings=settings)
    rephraser: object | None = (
        ClaudeRephraser(client=client, settings=settings)
        if settings.rephrase_enabled
        else None
    )

    if scenario.classifier_double is not None:
        classifier = _FixedClassifier(CLASSIFIER_DOUBLES[scenario.classifier_double])
    if scenario.provider_outage == "classifier":
        classifier = _OutageStage()
    elif scenario.provider_outage == "generator":
        generator = _OutageStage()
    elif scenario.provider_outage == "rephraser":
        rephraser = _OutageStage()

    counters = {
        "classifier": _CountingStage(classifier),
        "generator": _CountingStage(generator),
    }
    if rephraser is not None:
        counters["rephraser"] = _CountingStage(rephraser)
    return (
        AgentService(
            profile=profile,
            classifier=counters["classifier"],
            generator=counters["generator"],
            rephraser=counters.get("rephraser"),
        ),
        counters,
    )


def check_scenario(
    scenario: EvalScenario,
    response: Any,
    counters: dict[str, _CountingStage],
) -> list[str]:
    """Return every contract violation for one executed scenario, most specific first."""
    trace = response.trace
    answer = response.answer
    failures: list[str] = []

    if scenario.expected_outcome == "blocked":
        if trace.guardrail_input != "blocked":
            failures.append(f"expected a guardrail block, got {trace.guardrail_input!r}")
    elif scenario.expected_outcome == "not_found":
        if trace.grounding_status != "profile_missing":
            failures.append(f"expected profile_missing, got {trace.grounding_status!r}")
    elif scenario.expected_outcome == "clarify":
        if trace.grounding_status != "clarification":
            failures.append(f"expected clarification, got {trace.grounding_status!r}")
    elif trace.guardrail_input == "blocked" or trace.grounding_status in {
        "profile_missing",
        "clarification",
    }:
        failures.append(f"expected an answer, got {trace.grounding_status!r}")

    if scenario.expected_answer_mode and trace.answer_mode != scenario.expected_answer_mode:
        failures.append(
            f"answer_mode {trace.answer_mode!r} != {scenario.expected_answer_mode!r}"
        )
    if (
        scenario.expected_rendering_mode
        and trace.rendering_mode != scenario.expected_rendering_mode
    ):
        failures.append(
            f"rendering_mode {trace.rendering_mode!r} != {scenario.expected_rendering_mode!r}"
        )
    if scenario.expected_topic and trace.answer_topic != scenario.expected_topic:
        failures.append(f"answer_topic {trace.answer_topic!r} != {scenario.expected_topic!r}")

    # Tool scope, not merely tool presence: the right tool on the right topic.
    if scenario.tool_required and trace.tool_name is None:
        failures.append("expected a tool call, none was made")
    if scenario.expected_tool and trace.tool_name != scenario.expected_tool:
        failures.append(f"tool_name {trace.tool_name!r} != {scenario.expected_tool!r}")

    if scenario.expected_fact_ids:
        expected, actual = set(scenario.expected_fact_ids), set(trace.selected_fact_ids)
        if expected != actual:
            failures.append(
                f"fact scope missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
    if scenario.expected_source_ids:
        expected, actual = set(scenario.expected_source_ids), set(trace.selected_source_ids)
        if expected != actual:
            failures.append(
                f"source scope missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
    forbidden_facts = set(scenario.forbidden_fact_ids) & set(trace.selected_fact_ids)
    if forbidden_facts:
        failures.append(f"forbidden facts selected: {sorted(forbidden_facts)}")
    forbidden_sources = set(scenario.forbidden_source_ids) & set(trace.selected_source_ids)
    if forbidden_sources:
        failures.append(f"forbidden sources selected: {sorted(forbidden_sources)}")

    missing_tokens = [t for t in scenario.required_tokens if not token_present(t, answer)]
    if missing_tokens:
        failures.append(f"missing required tokens: {missing_tokens}")
    present_forbidden = [t for t in scenario.forbidden_tokens if token_present(t, answer)]
    if present_forbidden:
        failures.append(f"forbidden tokens present: {present_forbidden}")

    # An answer that may not infer must be reconstructed from canonical facts; only a
    # transformed answer carries wording the provider chose. A guardrail block or
    # out-of-scope redirect never reaches the answer plan at all (D-033), so it has no
    # rendering mode and cannot have inferred anything.
    if (
        not scenario.inference_permitted
        and trace.rendering_mode is not None
        and trace.rendering_mode not in DETERMINISTIC_RENDERING_MODES
    ):
        failures.append(
            f"inference is not permitted but rendering_mode is {trace.rendering_mode!r}"
        )

    if scenario.max_words is not None and len(answer.split()) > scenario.max_words:
        failures.append(f"{len(answer.split())} words exceeds {scenario.max_words}")
    if scenario.max_sentences is not None:
        from src.agent.rephrase import count_sentences

        sentences = count_sentences(answer)
        if sentences > scenario.max_sentences:
            failures.append(f"{sentences} sentences exceeds {scenario.max_sentences}")

    if scenario.max_generation_calls is not None:
        made = sum(
            counter.calls for name, counter in counters.items() if name != "classifier"
        )
        if made > scenario.max_generation_calls:
            failures.append(
                f"{made} generation/rephrase calls exceed {scenario.max_generation_calls}"
            )

    if scenario.expected_language and getattr(response.state, "response_language", None) not in {
        None,
        scenario.expected_language,
    }:
        failures.append(
            f"response_language {response.state.response_language!r} != "
            f"{scenario.expected_language!r}"
        )
    return failures


def execute_scenarios(
    scenarios: list[EvalScenario], settings: Settings
) -> list[dict[str, object]]:
    """Run the fixed matrix through the configured agent, retaining only safe metadata."""
    if settings.anthropic_api_key is None:
        raise GenerationUnavailableError(
            "ANTHROPIC_API_KEY is required for an executed evaluation run"
        )
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        service, counters = build_service(scenario, settings)
        started_at = perf_counter()
        response = service.respond(
            scenario.message,
            history=[item.model_dump() for item in scenario.history],
        )
        failures = check_scenario(scenario, response, counters)
        results.append(
            {
                "id": scenario.id,
                "passed": not failures,
                "failures": failures,
                "answer_mode": response.trace.answer_mode,
                "rendering_mode": response.trace.rendering_mode,
                "grounding_status": response.trace.grounding_status,
                "informativeness_outcome": response.trace.informativeness_outcome,
                "selected_fact_ids": response.trace.selected_fact_ids,
                "latency_ms": round((perf_counter() - started_at) * 1_000),
            }
        )
    return results


def main() -> int:
    """Validate scenario structure, and optionally execute the matrix against the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("eval/scenarios.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    arguments = parser.parse_args()
    scenarios = load_scenarios(arguments.scenarios)
    if not arguments.execute:
        print(
            json.dumps(
                {"scenarios_validated": len(scenarios), "status": "ready_for_model_run"}
            )
        )
        return 0
    settings = Settings()
    results = execute_scenarios(scenarios, settings)
    report = {
        "model": settings.model_name,
        "scenario_count": len(scenarios),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "passed": report["passed"],
                "failed": [item["id"] for item in results if not item["passed"]],
                "output": str(arguments.output),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
