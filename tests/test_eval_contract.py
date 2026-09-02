"""Contract tests for the evaluator itself.

A matrix that passes proves nothing unless the checker can fail. Each test here feeds
the checker a response that violates exactly one declared expectation and asserts that
the violation is reported, so a green evaluation run is evidence rather than habit.
"""

import json
from pathlib import Path

import pytest

from eval.run_eval import EvalScenario, check_scenario, load_scenarios
from src.agent.contracts import AgentResponse, AgentTrace, ConversationState


def _response(**trace_fields: object) -> AgentResponse:
    """Build a passing response, then let each test spoil one field."""
    defaults: dict[str, object] = {
        "answer": "Marco built the Security Console.",
        "answer_mode": "direct",
        "rendering_mode": "canonical",
        "answer_topic": "experience",
        "tool_name": "filter_experience",
        "grounding_status": "fact_rendered",
        "selected_fact_ids": ["fact:a"],
        "selected_source_ids": ["source:a"],
    }
    answer = str(trace_fields.pop("answer", defaults.pop("answer")))
    defaults.update(trace_fields)
    return AgentResponse(
        answer=answer,
        trace=AgentTrace(**defaults),  # type: ignore[arg-type]
        state=ConversationState(response_language="en"),
    )


def _scenario(**fields: object) -> EvalScenario:
    base: dict[str, object] = {
        "id": "probe",
        "message": "What security-related work has Marco done?",
        "expected_outcome": "answer",
        "inference_permitted": False,
    }
    base.update(fields)
    return EvalScenario.model_validate(base)


class _Counter:
    def __init__(self, calls: int) -> None:
        self.calls = calls


def test_a_conforming_response_reports_no_failures() -> None:
    """The checker must be satisfiable, or every other test here proves nothing."""
    scenario = _scenario(
        expected_answer_mode="direct",
        expected_rendering_mode="canonical",
        expected_topic="experience",
        expected_tool="filter_experience",
        expected_fact_ids=["fact:a"],
        expected_source_ids=["source:a"],
        required_tokens=["Security Console"],
        max_generation_calls=0,
    )

    failures = check_scenario(
        scenario, _response(), {"classifier": _Counter(1), "generator": _Counter(0)}
    )

    assert failures == []


def test_an_extra_selected_fact_fails_the_exact_scope() -> None:
    """A superset was previously tolerated; irrelevant extras must now be rejected."""
    scenario = _scenario(expected_fact_ids=["fact:a"])

    failures = check_scenario(
        scenario,
        _response(selected_fact_ids=["fact:a", "fact:unrelated"]),
        {"classifier": _Counter(1)},
    )

    assert any("extra=['fact:unrelated']" in failure for failure in failures)


def test_a_missing_selected_fact_fails_the_exact_scope() -> None:
    scenario = _scenario(expected_fact_ids=["fact:a", "fact:b"])

    failures = check_scenario(scenario, _response(), {"classifier": _Counter(1)})

    assert any("missing=['fact:b']" in failure for failure in failures)


def test_the_wrong_tool_fails_even_though_a_tool_ran() -> None:
    """`tool_required` alone could not tell a correct tool from any tool."""
    scenario = _scenario(expected_tool="filter_experience", tool_required=True)

    failures = check_scenario(
        scenario, _response(tool_name="query_profile"), {"classifier": _Counter(1)}
    )

    assert any("tool_name" in failure for failure in failures)


def test_inference_policy_is_enforced_not_merely_parsed() -> None:
    """A transformed answer is the only mode whose wording the provider chose."""
    scenario = _scenario(inference_permitted=False)

    failures = check_scenario(
        scenario, _response(rendering_mode="transformed"), {"classifier": _Counter(1)}
    )

    assert any("inference is not permitted" in failure for failure in failures)


def test_a_guardrail_boundary_without_a_rendering_mode_is_not_an_inference() -> None:
    """A block never reaches the answer plan, so it cannot have inferred anything."""
    scenario = _scenario(expected_outcome="blocked", inference_permitted=False)

    failures = check_scenario(
        scenario,
        _response(guardrail_input="blocked", rendering_mode=None, answer_mode=None),
        {"classifier": _Counter(0)},
    )

    assert failures == []


@pytest.mark.parametrize(
    ("field", "value", "answer", "fragment"),
    [
        ("max_words", 3, "one two three four five", "words exceeds"),
        ("max_sentences", 1, "One. Two. Three.", "sentences exceeds"),
    ],
)
def test_size_budgets_are_enforced(
    field: str, value: int, answer: str, fragment: str
) -> None:
    scenario = _scenario(**{field: value}, inference_permitted=True)

    failures = check_scenario(scenario, _response(answer=answer), {"classifier": _Counter(1)})

    assert any(fragment in failure for failure in failures)


def test_a_generation_call_on_a_deterministic_scenario_fails() -> None:
    """Direct answers must skip generation, and the runner must be able to prove it."""
    scenario = _scenario(max_generation_calls=0)

    failures = check_scenario(
        scenario,
        _response(),
        {"classifier": _Counter(1), "generator": _Counter(1)},
    )

    assert any("generation/rephrase calls exceed" in failure for failure in failures)


def test_a_forbidden_token_fails_and_a_classifier_call_does_not() -> None:
    scenario = _scenario(forbidden_tokens=["Google"], max_generation_calls=0)

    failures = check_scenario(
        scenario,
        _response(answer="Marco worked at Google."),
        {"classifier": _Counter(1), "generator": _Counter(0)},
    )

    assert any("forbidden tokens present" in failure for failure in failures)
    assert not any("generation/rephrase" in failure for failure in failures)


def test_the_shipped_matrix_covers_every_required_pairing() -> None:
    """Spec 03 fixes the matrix; drift in either direction should be visible here."""
    scenarios = load_scenarios(Path("eval/scenarios.json"))
    ids = {scenario.id for scenario in scenarios}

    for stem in (
        "start-date",
        "broad-experience",
        "broad-projects",
        "security-work",
        "named-technology",
        "experience-summary",
        "project-summary",
        "outage-synthesis",
    ):
        assert {f"{stem}-en", f"{stem}-es"} <= ids, stem
    assert {"impact-en", "impact-es"} <= ids
    assert {"outage-direct-en", "outage-direct-es"} <= ids
    assert {
        "double-companies-projects",
        "double-companies-experience",
        "double-companies-start-date",
        "double-summary-security",
        "double-skills-technology",
    } <= ids


def test_every_shipped_scenario_declares_an_enforceable_contract() -> None:
    """A scenario with no assertion beyond its outcome would pass without proving anything."""
    for scenario in load_scenarios(Path("eval/scenarios.json")):
        if scenario.expected_outcome != "answer":
            continue
        assert (
            scenario.expected_fact_ids
            or scenario.required_tokens
            or scenario.forbidden_tokens
        ), scenario.id


def test_scenarios_file_is_valid_json_and_uniquely_identified() -> None:
    raw = json.loads(Path("eval/scenarios.json").read_text(encoding="utf-8"))
    ids = [item["id"] for item in raw]
    assert len(ids) == len(set(ids))
