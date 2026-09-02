"""Issue #15: theme questions may span topics without making `AnswerPlan.topic` plural.

`search_resume` and `_synthesis_selection` pre-filter `fact.topic == topic` before
ranking, so a question about a cross-cutting domain ("What is his AI experience?")
cannot reach evidence that lives under a different topic (Sybil is `projects`,
the multi-agent ISV work is `experience`). This group pins the span policy:

- a profile-derived domain theme (currently "AI") may draw evidence from
  `projects` and `experience` together;
- `AnswerPlan.topic` stays scalar (the dominant topic), and a new
  `evidence_topics` lists every topic the selected facts belong to;
- date and single-field questions never span.
"""

import pytest

from src.agent.answer_planning import AnswerPlanner
from src.agent.contracts import GenerationUnavailableError, Intent, IntentDecision
from src.agent.orchestrator import AgentService
from src.models.profile import load_profile
from src.tools.profile_tools import SearchResumeArguments, search_resume

SYBIL_PROJECT = "fact:project:proj-sybil"
ISV_MODULE = "fact:experience:exp-global-payments.highlight:hl-isv-module"


@pytest.fixture
def profile():
    return load_profile("data/profile.json")


@pytest.fixture
def planner(profile):
    return AnswerPlanner(profile)


class StubClassifier:
    def __init__(self, decision: IntentDecision) -> None:
        self._decision = decision

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        return self._decision


class ProviderOutage:
    def classify(self, message: str, history: list[object]) -> IntentDecision:
        raise GenerationUnavailableError("provider unavailable")

    def generate(self, **_: object) -> object:
        raise AssertionError("must not call the generator")

    def rephrase(self, **_: object) -> object:
        raise AssertionError("must not call the rephraser")


def test_answer_plan_reports_evidence_topics_defaulting_to_its_scalar_topic(planner):
    """Every plan carries `evidence_topics`; an ordinary plan lists only its own topic."""
    plan = planner.explicit_direct_plan("Has Marco worked with FAISS?")

    assert plan is not None
    assert plan.topic == "projects"
    assert plan.evidence_topics == ["projects"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What is his AI experience?", "ai"),
        ("Tell me about Marco's AI work", "ai"),
        ("¿Cuál es su experiencia en inteligencia artificial?", "ai"),
        ("¿Cuál es su experiencia en IA?", "ai"),
        ("Has Marco worked with FAISS?", None),
        ("What security-related work has Marco done?", None),
        ("Summarize Marco's experience.", None),
        ("¿En qué proyectos ha trabajado Marco?", None),
    ],
)
def test_named_domain_recognizes_the_profile_ai_domain_only(planner, message, expected):
    """A profile-derived domain theme is distinct from a specific named technology."""
    assert planner.named_domain(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "What is his AI experience?",
        "¿Cuál es su experiencia en IA?",
    ],
)
def test_ai_theme_question_draws_evidence_from_projects_and_experience(planner, message):
    """The AI question reaches Sybil (projects) and the ISV work (experience), EN and ES."""
    plan = planner.explicit_direct_plan(message)

    assert plan is not None
    assert plan.topic in {"projects", "experience"}
    assert plan.evidence_topics == ["experience", "projects"]
    assert SYBIL_PROJECT in plan.selected_fact_ids
    assert ISV_MODULE in plan.selected_fact_ids


def test_ai_theme_selection_adds_no_fact_that_does_not_name_the_domain(planner):
    """Spanning keeps only genuine domain matches, never a parent/sibling filler fact."""
    plan = planner.explicit_direct_plan("What is his AI experience?")

    assert plan is not None
    # The employer parent record and the caching/security highlights are not AI work.
    assert "fact:experience:exp-global-payments" not in plan.selected_fact_ids
    assert (
        "fact:experience:exp-global-payments.highlight:hl-performance"
        not in plan.selected_fact_ids
    )


@pytest.mark.parametrize(
    "message",
    [
        "Since when has Marco worked at Global Payments?",
        "¿Desde cuándo trabaja Marco en Global Payments?",
    ],
)
def test_date_questions_never_span_topics(planner, message):
    """A single employment date is answered from one record; it never spans."""
    plan = planner.explicit_direct_plan(message)

    assert plan is not None
    assert plan.requested_field in {"start_date", "end_date", "current"}
    assert plan.evidence_topics == ["experience"]


def test_specific_named_technology_stays_scoped_to_its_one_topic(planner):
    """FAISS appears only in a Sybil highlight, so its answer never spans."""
    plan = planner.explicit_direct_plan("Has Marco worked with FAISS?")

    assert plan is not None
    assert plan.evidence_topics == ["projects"]


def test_summarising_the_ai_domain_may_reach_a_project_from_the_experience_topic(planner):
    """`Summarize his AI experience` keeps `topic` scalar but its evidence spans."""
    tool_result = search_resume(
        planner._profile,
        SearchResumeArguments(query="Summarize his AI experience."),
    )
    plan = planner.plan_from_tool("Summarize his AI experience.", tool_result)

    assert plan.synthesis_dimension == "summary"
    assert plan.topic in {"experience", "projects"}
    assert set(plan.evidence_topics) == {"experience", "projects"}
    assert SYBIL_PROJECT in plan.selected_fact_ids


def test_plain_experience_summary_without_a_domain_stays_within_its_topic(planner):
    """A domain word is what unlocks spanning; a bare summary is unchanged."""
    tool_result = search_resume(
        planner._profile, SearchResumeArguments(query="Summarize Marco's experience.")
    )
    plan = planner.plan_from_tool("Summarize Marco's experience.", tool_result)

    assert plan.topic == "experience"
    assert plan.evidence_topics == ["experience"]
    assert SYBIL_PROJECT not in plan.selected_fact_ids


def test_full_turn_for_the_ai_theme_question_delivers_both_records_and_a_scalar_topic():
    """End to end: the answer names Sybil and the multi-agent work; the trace spans."""
    service = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=StubClassifier(
            IntentDecision(intent=Intent.SEARCH_QUERY, confidence=0.95)
        ),
        generator=ProviderOutage(),
        rephraser=ProviderOutage(),
    )

    response = service.respond("What is his AI experience?", history=[])

    assert response.trace.answer_topic in {"projects", "experience"}
    assert response.trace.answer_topic in response.trace.evidence_topics
    assert set(response.trace.evidence_topics) == {"experience", "projects"}
    assert "Sybil" in response.answer
    assert "multi-agent" in response.answer or "ISV" in response.answer


def test_evidence_topics_are_projected_into_the_trace_fields(planner):
    """`plan_trace_fields` carries `evidence_topics` so every route reports it."""
    from src.agent.answer_planning import plan_trace_fields

    plan = planner.explicit_direct_plan("What is his AI experience?")
    fields = plan_trace_fields(plan, "canonical")

    assert fields["evidence_topics"] == ["experience", "projects"]


def test_a_spanning_turn_records_every_evidence_topic_in_the_discourse_record():
    """The conversation knows it touched both topics, not just the dominant one."""
    service = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=StubClassifier(
            IntentDecision(intent=Intent.SEARCH_QUERY, confidence=0.95)
        ),
        generator=ProviderOutage(),
        rephraser=ProviderOutage(),
    )

    response = service.respond("What is his AI experience?", history=[])

    assert response.state is not None
    assert set(response.state.discussed_topics) == {"experience", "projects"}
