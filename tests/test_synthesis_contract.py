"""Behavioral contract for concise, source-bounded synthesis answers."""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from src.agent.answer_planning import AnswerPlanner
from src.agent.contracts import (
    Claim,
    ClaimKind,
    GeneratedResponse,
    GenerationUnavailableError,
    Intent,
    IntentDecision,
    MAX_SYNTHESIS_FACTS,
    MAX_SYNTHESIS_PROPOSITIONS,
    SynthesisProposition,
    SynthesisTransformation,
)
from src.agent.orchestrator import AgentService
from src.models.profile import load_profile
from src.tools.profile_tools import (
    ProfileSummaryPlan,
    ResumeFact,
    ResumeSearchResult,
    SummarizeProfileArguments,
    fact_display_text,
    summarize_profile,
)


class SynthesisClassifier:
    """Stable classifier: the planner, not this double, owns synthesis scope."""

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        return IntentDecision(
            intent=Intent.SUMMARY_REQUEST,
            confidence=0.98,
            audience="recruiter",
        )


class GeneratorMustNotRun:
    """Configured synthesis transformation uses only the selected-fact rephraser."""

    def generate(self, **_: object) -> GeneratedResponse:
        raise AssertionError("the general generator must not run for bounded synthesis")


@dataclass(frozen=True)
class SynthesisCase:
    message: str
    dimension: str
    topic: str
    language: str
    candidate: str


CASES = (
    SynthesisCase(
        "Summarize Marco's experience.",
        "summary",
        "experience",
        "en",
        (
            "Marco works as a Jr. .NET Developer at Global Payments and implemented "
            "caching, resolving availability bottlenecks."
        ),
    ),
    SynthesisCase(
        "Resume la experiencia de Marco.",
        "summary",
        "experience",
        "es",
        (
            "Marco trabaja como Jr. .NET Developer en Global Payments e implementó "
            "caching, resolviendo cuellos de botella de disponibilidad."
        ),
    ),
    SynthesisCase(
        "Summarize the projects Marco has worked on.",
        "summary",
        "projects",
        "en",
        (
            "Marco built Sybil, a Python RAG system with a hybrid retrieval pipeline "
            "combining FAISS and SQLite FTS5."
        ),
    ),
    SynthesisCase(
        "What impact did Marco's work have?",
        "impact",
        "experience",
        "en",
        (
            "Marco collaborated on an ISV module, beating delivery expectations, and "
            "implemented caching, resolving availability bottlenecks."
        ),
    ),
    SynthesisCase(
        "¿Qué impacto tuvo el trabajo de Marco?",
        "impact",
        "experience",
        "es",
        (
            "Marco colaboró en la entrega de un módulo ISV, superando las expectativas "
            "de plazo, e implementó caching, resolviendo cuellos de botella de disponibilidad."
        ),
    ),
)


class RecordingTransformation:
    """Return one contract candidate and record the exact evidence boundary."""

    def __init__(self, candidate: str) -> None:
        self._candidate = candidate
        self.calls: list[dict[str, object]] = []

    def rephrase(
        self,
        *,
        message: str,
        facts: list[ResumeFact],
        language: str,

        feedback: str | None = None,
    ) -> SynthesisTransformation:
        self.calls.append({"message": message, "facts": facts, "language": language})
        return SynthesisTransformation(
            propositions=[
                SynthesisProposition(
                    text=self._candidate,
                    fact_ids=[fact.fact_id for fact in facts],
                )
            ],
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.dimension}-{case.language}-{case.topic}")
def test_synthesis_contract_selects_bounded_evidence_and_compresses_it(
    case: SynthesisCase,
) -> None:
    profile = load_profile("data/profile.json")
    transformer = RecordingTransformation(case.candidate)
    service = AgentService(
        profile=profile,
        classifier=SynthesisClassifier(),
        generator=GeneratorMustNotRun(),
        rephraser=transformer,
    )

    response = service.respond(case.message, history=[])

    assert response.trace.answer_mode == "synthesis"
    assert response.trace.synthesis_dimension == case.dimension
    assert response.trace.answer_topic == case.topic
    assert response.trace.rendering_mode == "transformed"
    assert response.trace.transformation_outcome == "accepted"
    assert response.trace.final_sentence_count <= 3
    assert response.trace.final_word_count <= 75
    assert response.trace.final_sentence_count >= 1
    assert response.trace.final_word_count == len(response.answer.split())
    assert response.answer == case.candidate
    assert len(transformer.calls) == 1
    call = transformer.calls[0]
    assert call["language"] == case.language
    selected_facts = call["facts"]
    assert isinstance(selected_facts, list)
    assert 1 < len(selected_facts) <= 3
    assert [fact.fact_id for fact in selected_facts] == response.trace.selected_fact_ids
    assert list(dict.fromkeys(fact.source_id for fact in selected_facts)) == (
        response.trace.selected_source_ids
    )
    assert len(response.trace.selected_fact_ids) == len(set(response.trace.selected_fact_ids))
    if case.dimension == "impact":
        selected_text = " ".join(fact.text.casefold() for fact in selected_facts)
        assert any(
            outcome in selected_text
            for outcome in ("ahead", "beating", "resolved", "reduced", "improve")
        )


@pytest.mark.parametrize(
    ("english", "spanish"),
    [
        ("Summarize Marco's experience.", "Resume la experiencia de Marco."),
        ("What impact did Marco's work have?", "¿Qué impacto tuvo el trabajo de Marco?"),
    ],
)
def test_equivalent_synthesis_requests_select_the_same_ranked_scope(
    english: str,
    spanish: str,
) -> None:
    """Same dimension, same topic, same ranking; Spanish takes a prefix of it (D-036)."""
    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)
    tool_plan = summarize_profile(
        profile,
        SummarizeProfileArguments(audience="recruiter"),
    )

    english_plan = planner.plan_from_tool(english, tool_plan)
    spanish_plan = planner.plan_from_tool(spanish, tool_plan)

    assert english_plan.synthesis_dimension == spanish_plan.synthesis_dimension
    assert english_plan.topic == spanish_plan.topic
    assert english_plan.scope == spanish_plan.scope
    assert english_plan.requested_field == spanish_plan.requested_field
    # Spanish carries the same evidence in the same order, cut one earlier, so the two
    # answers can never disagree about what matters most or cite something the other
    # would not have selected.
    assert spanish_plan.selected_fact_ids == english_plan.selected_fact_ids[
        : len(spanish_plan.selected_fact_ids)
    ]
    assert len(spanish_plan.selected_fact_ids) < len(english_plan.selected_fact_ids)


@pytest.mark.parametrize(
    ("message", "mode", "candidate"),
    [
        ("Summarize Marco's experience.", "outage", None),
        (
            "Resume la experiencia de Marco.",
            "rejection",
            "Marco led Google to $9M in revenue using Kubernetes.",
        ),
    ],
)
def test_synthesis_outage_or_rejection_returns_one_concise_canonical_boundary(
    message: str,
    mode: str,
    candidate: str | None,
) -> None:
    class BoundaryTransformation:
        def __init__(self) -> None:
            self.calls = 0
            self.facts: list[ResumeFact] = []

        def rephrase(self, **kwargs: object) -> SynthesisTransformation:
            self.calls += 1
            facts = kwargs["facts"]
            assert isinstance(facts, list)
            self.facts = facts
            if mode == "outage":
                raise GenerationUnavailableError("provider unavailable")
            assert candidate is not None
            return SynthesisTransformation(
                propositions=[
                    SynthesisProposition(
                        text=candidate,
                        fact_ids=[fact.fact_id for fact in self.facts],
                    )
                ],
            )

    transformer = BoundaryTransformation()
    service = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=GeneratorMustNotRun(),
        rephraser=transformer,
    )

    response = service.respond(message, history=[])

    # An outage is terminal; a rejection earns exactly one corrective attempt.
    assert transformer.calls == (1 if mode == "outage" else 2)
    assert response.trace.answer_mode == "synthesis"
    assert response.trace.rendering_mode == "canonical_fallback"
    assert response.trace.transformation_outcome.startswith(
        "unavailable" if mode == "outage" else "rejected"
    )
    assert response.trace.fallback_reason
    assert response.trace.final_sentence_count <= 3
    assert response.trace.final_word_count <= 75
    assert response.trace.final_word_count == len(response.answer.split())
    assert "Google" not in response.answer
    assert "Kubernetes" not in response.answer
    assert "$9M" not in response.answer
    assert " led " not in f" {response.answer.casefold()} "
    selected = transformer.calls and response.trace.selected_fact_ids
    assert selected
    rendered_selected = [
        fact_display_text(fact, "es" if message.startswith("Resume") else "en")
        for fact in transformer.facts
    ]
    assert sum(text in response.answer for text in rendered_selected) < len(rendered_selected)


@pytest.mark.parametrize(
    ("message", "candidate", "forbidden"),
    [
        (
            "Summarize Marco's experience.",
            "Marco implemented caching with kubernetes.",
            "kubernetes",
        ),
        (
            "Summarize Marco's experience.",
            "Google improved availability.",
            "Google",
        ),
        (
            "Summarize Marco's experience.",
            "9000 users benefited from caching.",
            "9000",
        ),
        (
            "Summarize Marco's experience.",
            "Marco designed caching for availability.",
            "designed",
        ),
        (
            "Resume la experiencia de Marco.",
            "Marco implemented caching that resolved availability bottlenecks.",
            "implemented",
        ),
        (
            "What impact did Marco's work have?",
            "Marco implemented caching, boosting productivity.",
            "productivity",
        ),
        (
            "Summarize Marco's experience.",
            (
                "Marco implemented caching. it resolved bottlenecks. "
                "it improved availability. it supported security."
            ),
            "it supported security",
        ),
    ],
)
def test_adversarial_synthesis_never_crosses_the_public_boundary(
    message: str,
    candidate: str,
    forbidden: str,
) -> None:
    class AdversarialTransformation:
        def rephrase(self, **kwargs: object) -> SynthesisTransformation:
            facts = kwargs["facts"]
            assert isinstance(facts, list)
            return SynthesisTransformation(
                propositions=[
                    SynthesisProposition(
                        text=candidate,
                        fact_ids=[fact.fact_id for fact in facts],
                    )
                ],
            )

    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=GeneratorMustNotRun(),
        rephraser=AdversarialTransformation(),
    ).respond(message, history=[])

    assert response.trace.rendering_mode == "canonical_fallback"
    assert response.trace.transformation_outcome.startswith("rejected:")
    assert forbidden.casefold() not in response.answer.casefold()
    assert response.trace.final_sentence_count <= 3
    assert response.trace.final_word_count <= 75


def test_general_synthesis_provider_receives_only_the_planned_fact_projection() -> None:
    class PayloadSpy:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, **kwargs: object) -> GeneratedResponse:
            self.calls.append(kwargs)
            raise GenerationUnavailableError("provider unavailable")

    spy = PayloadSpy()
    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=spy,
    ).respond("Summarize Marco's experience.", history=[])

    assert len(spy.calls) == 1
    payload = spy.calls[0]
    allowed_facts = payload["allowed_facts"]
    assert isinstance(allowed_facts, list)
    assert [fact.fact_id for fact in allowed_facts] == response.trace.selected_fact_ids
    projected = payload["tool_result"]
    assert isinstance(projected, ProfileSummaryPlan)
    assert projected.fact_ids == response.trace.selected_fact_ids
    assert "hl-stakeholder-coord" not in projected.model_dump_json()
    assert response.trace.rendering_mode == "canonical_fallback"


def test_general_synthesis_verifies_each_claim_against_its_own_citations() -> None:
    class WrongSelectedCitation:
        def generate(self, **kwargs: object) -> GeneratedResponse:
            facts = kwargs["allowed_facts"]
            assert isinstance(facts, list)
            parent = next(
                fact for fact in facts if fact.fact_id == "fact:experience:exp-global-payments"
            )
            text = "Marco implemented caching."
            return GeneratedResponse(
                text=text,
                claims=[
                    Claim(
                        text=text,
                        kind=ClaimKind.DIRECT,
                        fact_ids=[parent.fact_id],
                        source_ids=[parent.source_id],
                    )
                ],
            )

    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=WrongSelectedCitation(),
    ).respond("Summarize Marco's experience.", history=[])

    assert response.trace.rendering_mode == "canonical_fallback"
    assert response.trace.transformation_outcome.startswith("rejected:")


def test_the_contract_cannot_express_one_proposition_per_selected_fact() -> None:
    """A full selection has more facts than propositions, so a dump has no valid shape."""
    with pytest.raises(ValidationError):
        SynthesisTransformation(
            propositions=[
                SynthesisProposition(
                    text="Marco works as a Jr. .NET Developer at Global Payments.",
                    fact_ids=["fact:experience:exp-global-payments"],
                ),
                SynthesisProposition(
                    text="Marco implemented Redis and SQL caching.",
                    fact_ids=["fact:experience:exp-global-payments.highlight:hl-performance"],
                ),
                SynthesisProposition(
                    text="Marco collaborated in delivering an ISV module.",
                    fact_ids=["fact:experience:exp-global-payments.highlight:hl-isv-module"],
                ),
            ]
        )


def test_a_concise_conclusion_rejects_more_than_one_supporting_example() -> None:
    """Two propositions still overrun when one of them carries a second example."""

    class UncompressedTransformation:
        def rephrase(self, **_: object) -> SynthesisTransformation:
            return SynthesisTransformation(
                propositions=[
                    SynthesisProposition(
                        text="Marco works as a Jr. .NET Developer at Global Payments.",
                        fact_ids=["fact:experience:exp-global-payments"],
                    ),
                    SynthesisProposition(
                        text=(
                            "Marco implemented Redis and SQL caching. Marco resolved "
                            "availability-affecting performance bottlenecks."
                        ),
                        fact_ids=[
                            "fact:experience:exp-global-payments.highlight:hl-performance"
                        ],
                    ),
                ]
            )

    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=GeneratorMustNotRun(),
        rephraser=UncompressedTransformation(),
    ).respond("What conclusion can you draw about Marco's experience?", history=[])

    assert response.trace.rendering_mode == "canonical_fallback"
    assert response.trace.transformation_outcome == "rejected:too_many_examples"


@pytest.mark.parametrize(
    ("message", "allowed_source_suffixes", "must_not_select"),
    [
        (
            "What impact did Marco's security work have?",
            ("hl-performance",),
            ("hl-isv-module", "hl-reusable-apis"),
        ),
        (
            "What impact did Marco's FAISS work have?",
            ("sybil-hl-hybrid",),
            ("hl-isv-module", "hl-performance", "hl-reusable-apis"),
        ),
    ],
)
def test_impact_selection_stays_inside_explicit_current_topic(
    message: str,
    allowed_source_suffixes: tuple[str, ...],
    must_not_select: tuple[str, ...],
) -> None:
    class Outage:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_: object) -> GeneratedResponse:
            self.calls += 1
            raise GenerationUnavailableError("provider unavailable")

    outage = Outage()
    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=outage,
    ).respond(message, history=[])

    assert response.trace.answer_mode == "synthesis"
    assert response.trace.rendering_mode == "canonical_fallback"
    assert response.trace.selected_source_ids
    assert all(
        any(source_id.endswith(suffix) for suffix in allowed_source_suffixes)
        for source_id in response.trace.selected_source_ids
    )
    assert not any(
        forbidden in source_id
        for source_id in response.trace.selected_source_ids
        for forbidden in must_not_select
    )
    if "FAISS" in message:
        assert outage.calls == 0
        assert response.trace.fallback_reason == "missing_explicit_impact"


@pytest.mark.parametrize(
    "message",
    [
        "What impact did Marco's work have?",
        "¿Qué impacto tuvo el trabajo de Marco?",
        "Compare Marco's projects.",
        "Compara los proyectos de Marco.",
    ],
)
def test_classifier_and_provider_outage_use_selected_canonical_fallback(message: str) -> None:
    class SharedOutage:
        def classify(self, message: str, history: list[object]) -> IntentDecision:
            raise GenerationUnavailableError("provider unavailable")

        def generate(self, **_: object) -> GeneratedResponse:
            raise GenerationUnavailableError("provider unavailable")

    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SharedOutage(),
        generator=SharedOutage(),
    ).respond(message, history=[])

    assert response.answer
    assert response.trace.answer_mode == "synthesis"
    assert response.trace.selected_fact_ids
    assert response.trace.rendering_mode == "canonical_fallback"
    assert response.trace.final_sentence_count <= 3
    assert response.trace.final_word_count <= 75


def test_bilingual_comparison_selects_equivalent_project_scope() -> None:
    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)
    tool_plan = summarize_profile(
        profile,
        SummarizeProfileArguments(audience="recruiter"),
    )

    english = planner.plan_from_tool("Compare Marco's projects.", tool_plan)
    spanish = planner.plan_from_tool("Compara los proyectos de Marco.", tool_plan)

    assert english.synthesis_dimension == spanish.synthesis_dimension == "comparison"
    assert english.topic == spanish.topic == "projects"
    # Same ranking, Spanish cut one earlier (D-036).
    assert spanish.selected_fact_ids == english.selected_fact_ids[
        : len(spanish.selected_fact_ids)
    ]


class MultiPropositionTransformation:
    """Return the aggregated shape a real provider emits: several mapped propositions."""

    def __init__(self, propositions: tuple[str, ...]) -> None:
        self._propositions = propositions
        self.facts: list[ResumeFact] = []

    def rephrase(
        self,
        *,
        message: str,
        facts: list[ResumeFact],
        language: str,

        feedback: str | None = None,
    ) -> SynthesisTransformation:
        self.facts = facts
        return SynthesisTransformation(
            propositions=[
                SynthesisProposition(text=text, fact_ids=[fact.fact_id])
                for text, fact in zip(self._propositions, facts, strict=False)
            ]
        )


def test_synthesis_accepts_propositions_without_a_duplicated_top_level_text() -> None:
    """A provider states the answer once; a second copy can only drift and be rejected."""
    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)
    plan = planner.plan_from_tool(
        "Summarize Marco's experience.",
        summarize_profile(profile, SummarizeProfileArguments(audience="recruiter")),
    )
    propositions = (
        "Marco works as a Jr. .NET Developer at Global Payments.",
        "He implemented caching, resolving availability bottlenecks.",
    )
    transformer = MultiPropositionTransformation(propositions)
    service = AgentService(
        profile=profile,
        classifier=SynthesisClassifier(),
        generator=GeneratorMustNotRun(),
        rephraser=transformer,
    )

    response = service.respond("Summarize Marco's experience.", history=[])

    assert response.trace.transformation_outcome == "accepted"
    assert response.trace.rendering_mode == "transformed"
    assert response.answer == " ".join(propositions)
    assert [fact.fact_id for fact in transformer.facts] == plan.selected_fact_ids


def test_a_full_selection_cannot_be_delivered_as_one_proposition_per_fact() -> None:
    """Aggregation is structural: fewer propositions than facts leaves no 1:1 mapping."""
    assert MAX_SYNTHESIS_PROPOSITIONS < MAX_SYNTHESIS_FACTS

    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)
    schema_maximum = SynthesisTransformation.model_fields["propositions"].metadata[-1].max_length
    assert schema_maximum == MAX_SYNTHESIS_PROPOSITIONS

    for message in (
        "Summarize Marco's experience.",
        "Resume la experiencia de Marco.",
        "Summarize the projects Marco has worked on.",
        "What impact did Marco's work have?",
        "¿Qué impacto tuvo el trabajo de Marco?",
    ):
        plan = planner.plan_from_tool(
            message,
            summarize_profile(profile, SummarizeProfileArguments(audience="recruiter")),
        )
        assert len(plan.selected_fact_ids) <= MAX_SYNTHESIS_FACTS, message


class CorrectableTransformation:
    """Reject the first attempt, then answer inside the gate once told what was wrong."""

    def __init__(self) -> None:
        self.feedback: list[str | None] = []

    def rephrase(self, **kwargs: object) -> SynthesisTransformation:
        feedback = kwargs.get("feedback")
        assert isinstance(feedback, (str, type(None)))
        self.feedback.append(feedback)
        if feedback is None:
            return SynthesisTransformation(
                propositions=[
                    SynthesisProposition(
                        text="Marco spearheaded a Kubernetes migration.",
                        fact_ids=["fact:experience:exp-global-payments"],
                    )
                ]
            )
        return SynthesisTransformation(
            propositions=[
                SynthesisProposition(
                    text="Marco works as a Jr. .NET Developer at Global Payments.",
                    fact_ids=["fact:experience:exp-global-payments"],
                )
            ]
        )


def test_a_rejected_transformation_earns_one_corrective_attempt() -> None:
    """The gate already names the defect; sending it back beats loosening the gate."""
    transformer = CorrectableTransformation()

    response = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=SynthesisClassifier(),
        generator=GeneratorMustNotRun(),
        rephraser=transformer,
    ).respond("Summarize Marco's experience.", history=[])

    assert transformer.feedback[0] is None
    assert transformer.feedback[1]
    assert response.trace.rendering_mode == "transformed"
    assert response.trace.transformation_outcome == "accepted_after_correction"


@pytest.mark.parametrize(
    ("message", "topic"),
    [
        ("Resume la experiencia de Marco.", "experience"),
        ("Resume los proyectos de Marco.", "projects"),
        ("Resume sus proyectos.", "projects"),
        ("Resúmeme la experiencia de Marco.", "experience"),
    ],
)
def test_spanish_summary_requests_are_recognized_beyond_one_fixed_phrase(
    message: str,
    topic: str,
) -> None:
    """"Resume" is a Spanish imperative verb; its object is not always "la experiencia"."""
    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)

    plan = planner.plan_from_tool(
        message,
        summarize_profile(profile, SummarizeProfileArguments(audience="recruiter")),
    )

    assert plan.synthesis_dimension == "summary"
    assert plan.topic == topic


def test_an_english_resume_noun_is_not_a_summary_request() -> None:
    """The same letters are a noun in English and must not trigger transformation."""
    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)

    plan = planner.plan_from_tool(
        "What does Marco's resume say about FAISS?",
        summarize_profile(profile, SummarizeProfileArguments(audience="recruiter")),
    )

    assert plan.synthesis_dimension is None
