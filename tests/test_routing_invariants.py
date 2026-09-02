"""Offline routing invariants: paraphrase families, perturbation, and system rules.

The 30-scenario matrix is a regression suite, and passing it proves the implementation
handles those 30 phrasings. It does not prove the hard gate's promise that reasonable
résumé questions are robustly handled, because a fixed matrix never varies the one
dimension real users vary constantly: surface form.

This group runs provider-free by construction (D-033) against `AnswerPlanner`,
`detect_response_language`, and the deterministic profile tools, with the classifier
stubbed. It therefore costs milliseconds and can run on every push, while the live
matrix stays small and covers what genuinely needs a provider: prose, containment,
and transformation.

Every family declares both the variants expected to resolve and the variants
deliberately out of scope, each with a reason. A variant count is only interpretable
if the denominator is stated.
"""

import unicodedata
from dataclasses import dataclass

import pytest

from src.agent.answer_planning import AnswerPlanner
from src.agent.contracts import (
    Claim,
    ClaimKind,
    GeneratedResponse,
    Intent,
    IntentDecision,
    InvalidStructuredOutputError,
)
from src.agent.orchestrator import AgentService
from src.models.profile import load_profile
from src.tools.profile_tools import (
    SearchResumeArguments,
    detect_response_language,
    search_resume,
)


@dataclass(frozen=True)
class ParaphraseFamily:
    """One semantic request expressed many ways, with its scope stated explicitly."""

    name: str
    dimension: str | None
    topic: str
    language: str
    resolves: tuple[str, ...]
    out_of_scope: tuple[tuple[str, str], ...] = ()


FAMILIES = (
    ParaphraseFamily(
        name="achievements-en",
        dimension="impact",
        topic="experience",
        language="en",
        resolves=(
            "What are Marco's achievements?",
            "What did Marco accomplish?",
            "What has Marco achieved?",
            "Marco's accomplishments",
            "Which achievements does Marco have?",
            "What did Marco accomplish in his work?",
            "Tell me Marco's achievements",
        ),
        out_of_scope=(
            (
                "What results has Marco produced?",
                "English 'results' is absent from the impact markers; only the Spanish "
                "'resultado' is present. A real gap, filed separately, not a phrasing "
                "this family claims to cover.",
            ),
        ),
    ),
    ParaphraseFamily(
        name="achievements-es",
        dimension="impact",
        topic="experience",
        language="es",
        resolves=(
            "Cuales son los logros de Marco?",
            "Que logros ha tenido marco?",
            "¿Qué logró Marco?",
            "Sus logros?",
            "¿Cuáles fueron sus logros?",
            "Logros de Marco",
            "¿Qué logró Marco en su trabajo?",
            "¿Qué logros consiguió Marco?",
        ),
    ),
    ParaphraseFamily(
        name="impact-en",
        dimension="impact",
        topic="experience",
        language="en",
        resolves=(
            "What impact did Marco's work have?",
            "What outcomes did Marco deliver?",
            "What was the impact of his work?",
        ),
    ),
    ParaphraseFamily(
        name="impact-es",
        dimension="impact",
        topic="experience",
        language="es",
        resolves=(
            "¿Qué impacto tuvo el trabajo de Marco?",
            "¿Qué resultados obtuvo Marco?",
            "¿Cuál fue el efecto de su trabajo?",
        ),
    ),
    ParaphraseFamily(
        name="experience-summary-en",
        dimension="summary",
        topic="experience",
        language="en",
        resolves=(
            "Summarize Marco's experience.",
            "Give me a summary of his experience",
            "Summarize his work history",
        ),
    ),
    ParaphraseFamily(
        name="experience-summary-es",
        dimension="summary",
        topic="experience",
        language="es",
        resolves=(
            "Resume la experiencia de Marco.",
            "Resumen de su experiencia",
            "Resume el trabajo de Marco",
        ),
        out_of_scope=(
            (
                "Resume su trayectoria laboral",
                "'trayectoria laboral' contains no experience topic token, so the "
                "topic falls back rather than resolving to employment.",
            ),
        ),
    ),
)

BILINGUAL_PAIRS = (
    ("achievements-en", "achievements-es"),
    ("impact-en", "impact-es"),
    ("experience-summary-en", "experience-summary-es"),
)

# The fixed bilingual UI release script from specs/03. Reproduced here as the
# canonical-answerable set, not to replace the manual gate, which stays unchanged.
RELEASE_SCRIPT = (
    "¿Desde cuándo trabaja Marco en Global Payments?",
    "Has Marco worked with FAISS?",
    "¿En qué proyectos ha trabajado Marco?",
    "What security-related work has Marco done?",
    "Dime acerca de la experiencia de Marco",
    "Summarize Marco's experience.",
    "Resume la experiencia de Marco.",
    "Summarize the projects Marco has worked on.",
    "What impact did Marco's work have?",
    "¿Qué impacto tuvo el trabajo de Marco?",
)

ALL_VARIANTS = tuple(
    dict.fromkeys(
        [variant for family in FAMILIES for variant in family.resolves]
        + [variant for family in FAMILIES for variant, _ in family.out_of_scope]
        + list(RELEASE_SCRIPT)
    )
)


def strip_accents(text: str) -> str:
    """Remove diacritics and Spanish opening punctuation, as users typing fast do."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return without_marks.replace("¿", "").replace("¡", "")


CLASSIFIER_DECISIONS = (
    IntentDecision(intent=Intent.SEARCH_QUERY, confidence=0.95),
    IntentDecision(intent=Intent.SUMMARY_REQUEST, confidence=0.95, audience="recruiter"),
    IntentDecision(
        intent=Intent.DIRECT_QUESTION,
        confidence=0.95,
        profile_field="companies",
    ),
)


class StubClassifier:
    """Return one fixed decision so families can vary classifier output offline."""

    def __init__(self, decision: IntentDecision = CLASSIFIER_DECISIONS[0]) -> None:
        self._decision = decision

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        return self._decision


class EmptyFactSetIsAFailure:
    """Generator double that fails loudly rather than reproducing the #6 defect."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: object) -> GeneratedResponse:
        self.calls += 1
        allowed_facts = kwargs.get("allowed_facts") or []
        if not allowed_facts:
            raise AssertionError("the generator was invoked with an empty fact set")
        return GeneratedResponse(
            text="Marco worked on the security console.",
            claims=[
                Claim(
                    text="Marco worked on the security console.",
                    kind=ClaimKind.DIRECT,
                    source_ids=["experience:exp-global-payments"],
                    evidence=["security console"],
                )
            ],
        )


def route(message: str) -> tuple[str | None, str, str]:
    """Resolve a message to its semantic routing, with no provider in the path."""
    profile = load_profile("data/profile.json")
    planner = AnswerPlanner(profile)
    tool_result = search_resume(profile, SearchResumeArguments(query=message))
    plan = planner.plan_from_tool(message, tool_result)
    return plan.synthesis_dimension, plan.topic, plan.language


def answer(
    message: str,
    decision: IntentDecision = CLASSIFIER_DECISIONS[0],
) -> object:
    """Run one full turn offline with a stubbed classifier and a strict generator."""
    service = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=StubClassifier(decision),
        generator=EmptyFactSetIsAFailure(),
    )
    return service.respond(message, history=[])


def routing_signature(response: object) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Return the evidence contract that classifier disagreement must not change."""
    trace = response.trace
    return trace.synthesis_dimension, trace.answer_topic, tuple(trace.selected_fact_ids)


def assert_stable_routing(
    signatures: list[tuple[str | None, str | None, tuple[str, ...]]],
) -> None:
    """Fail with every observed route when one semantic family diverges."""
    assert signatures
    assert all(signature == signatures[0] for signature in signatures), signatures


FAMILY_VARIANTS = [
    (family, variant) for family in FAMILIES for variant in family.resolves
]


@pytest.mark.parametrize(
    ("family", "variant"),
    FAMILY_VARIANTS,
    ids=[f"{family.name}::{variant}" for family, variant in FAMILY_VARIANTS],
)
def test_surface_form_does_not_change_semantic_routing(
    family: ParaphraseFamily, variant: str
) -> None:
    """Every declared variant of one request must reach the same dimension and topic."""
    dimension, topic, language = route(variant)

    assert dimension == family.dimension
    assert topic == family.topic
    assert language == family.language


@pytest.mark.parametrize(
    ("family", "variant"),
    FAMILY_VARIANTS,
    ids=[f"{family.name}::{variant}" for family, variant in FAMILY_VARIANTS],
)
def test_accent_removal_preserves_routing(
    family: ParaphraseFamily, variant: str
) -> None:
    """Typing without accents is normal input, not a different question."""
    assert route(strip_accents(variant)) == route(variant)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.name)
def test_out_of_scope_variants_are_declared_rather_than_silently_broken(
    family: ParaphraseFamily,
) -> None:
    """A variant this family does not cover must be stated, with its reason.

    If one starts resolving, this fails and the variant is promoted to `resolves`.
    An out-of-scope phrasing may be unresolved; it may never be unsafe.
    """
    for variant, reason in family.out_of_scope:
        assert reason
        dimension, topic, _ = route(variant)
        assert (dimension, topic) != (family.dimension, family.topic), (
            f"{variant!r} now resolves; promote it out of {family.name}'s out_of_scope"
        )
        response = answer(variant)
        assert response.answer


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.name)
def test_family_coverage_is_interpretable(family: ParaphraseFamily) -> None:
    """A variant count means nothing without a stated denominator."""
    assert len(family.resolves) >= 3
    assert not set(family.resolves) & {variant for variant, _ in family.out_of_scope}


@pytest.mark.parametrize(
    ("english_name", "spanish_name"),
    BILINGUAL_PAIRS,
    ids=[f"{english}~{spanish}" for english, spanish in BILINGUAL_PAIRS],
)
def test_equivalent_bilingual_families_resolve_alike(
    english_name: str, spanish_name: str
) -> None:
    """Equivalent requests must not diverge in dimension or topic across languages."""
    by_name = {family.name: family for family in FAMILIES}
    english, spanish = by_name[english_name], by_name[spanish_name]

    assert english.dimension == spanish.dimension
    assert english.topic == spanish.topic
    for variant in english.resolves:
        assert route(variant)[:2] == (english.dimension, english.topic)
    for variant in spanish.resolves:
        assert route(variant)[:2] == (spanish.dimension, spanish.topic)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family.name)
def test_classifier_disagreement_does_not_change_family_evidence(
    family: ParaphraseFamily,
) -> None:
    """Current-message semantics must outrank incompatible classifier tool choices."""
    signatures = [
        routing_signature(answer(variant, decision))
        for variant in family.resolves
        for decision in CLASSIFIER_DECISIONS
    ]

    assert_stable_routing(signatures)
    dimension, topic, selected_fact_ids = signatures[0]
    assert dimension == family.dimension
    assert topic == family.topic
    assert selected_fact_ids


@pytest.mark.parametrize("message", ALL_VARIANTS, ids=list(ALL_VARIANTS))
def test_no_variant_invokes_the_generator_with_an_empty_fact_set(message: str) -> None:
    """The #6 invariant, asserted across every phrasing this group knows."""
    response = answer(message)

    assert response.answer


@pytest.mark.parametrize("message", RELEASE_SCRIPT, ids=list(RELEASE_SCRIPT))
def test_canonical_answerable_questions_are_answered(message: str) -> None:
    """A release-script question has sufficient canonical facts; it must not deflect."""
    response = answer(message)

    assert response.trace.rendering_mode != "clarification"
    assert response.trace.selected_fact_ids
    assert response.trace.selection_path in {"primary", "recovery"}


@pytest.mark.parametrize("message", ALL_VARIANTS, ids=list(ALL_VARIANTS))
def test_accent_removal_preserves_the_detected_language(message: str) -> None:
    """Detection must read more than orthography, or every unaccented user gets English."""
    assert detect_response_language(strip_accents(message)) == (
        detect_response_language(message)
    )


def test_the_empty_fact_set_guard_is_not_vacuous() -> None:
    """The invariant above is only evidence if its detector actually fires."""
    with pytest.raises(AssertionError, match="empty fact set"):
        EmptyFactSetIsAFailure().generate(allowed_facts=[], message="x")


def test_routing_assertions_discriminate_between_families() -> None:
    """Equally, family assertions are only evidence if routing can differ at all."""
    assert route("What are Marco's achievements?")[:2] == ("impact", "experience")
    assert route("Summarize Marco's experience.")[:2] == ("summary", "experience")
    assert route("What are Marco's achievements?") != route("Summarize Marco's experience.")


def test_classifier_disagreement_detector_is_not_vacuous() -> None:
    """The classifier invariant is evidence only if unequal fact selections fail it."""
    with pytest.raises(AssertionError):
        assert_stable_routing(
            [
                ("impact", "experience", ("fact:one",)),
                ("impact", "experience", ("fact:two",)),
            ]
        )


def test_language_assertions_discriminate_between_languages() -> None:
    """And the language invariant is only evidence if both verdicts are reachable."""
    assert detect_response_language("Summarize Marco's experience.") == "en"
    assert detect_response_language("Resume la experiencia de Marco.") == "es"


# --- #19: the classifier-failure door, negative referents, and term leakage ---

UNROUTABLE_MESSAGES = (
    # The observed transcript turn that produced no response at all: a reference to
    # something the agent never said, matching no follow-up phrase and no marker.
    "The part where it says he worked at google",
    "La parte donde dice que trabajó en Google",
    "When you said he led the payments migration",
    # Deliberately unanchored input, which may be clarified but must never 503.
    "the part you mentioned",
    "that bit",
)

TERM_LEAKAGE_MESSAGES = (
    "Tell me about his projects",
    "Resume los proyectos de Marco.",
)


class ClassifierAlwaysFails:
    """Both classifier attempts return malformed structured output, as observed in #19."""

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        raise InvalidStructuredOutputError("classifier returned malformed JSON")


def answer_without_classifier(message: str) -> object:
    """Run one full turn with the classifier permanently unavailable."""
    service = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=ClassifierAlwaysFails(),
        generator=EmptyFactSetIsAFailure(),
    )
    return service.respond(message, history=[])


@pytest.mark.parametrize(
    "message",
    ALL_VARIANTS + UNROUTABLE_MESSAGES,
    ids=list(ALL_VARIANTS + UNROUTABLE_MESSAGES),
)
def test_classifier_failure_degrades_instead_of_reaching_the_503_door(message: str) -> None:
    """Every boundary degrades to a deterministic HTTP 200; this one must too."""
    response = answer_without_classifier(message)

    assert response.answer


@pytest.mark.parametrize("message", RELEASE_SCRIPT, ids=list(RELEASE_SCRIPT))
def test_classifier_failure_still_answers_canonical_questions(message: str) -> None:
    """A question with sufficient canonical facts must not deflect when the model dies."""
    response = answer_without_classifier(message)

    assert response.trace.rendering_mode != "clarification"
    assert response.trace.selected_fact_ids


def test_the_classifier_failure_double_actually_fails() -> None:
    """The invariant above is only evidence if the stub really raises."""
    with pytest.raises(InvalidStructuredOutputError):
        ClassifierAlwaysFails().classify("x", [])


@pytest.mark.parametrize(
    "message",
    ("The part where it says he worked at google", "La parte donde dice que trabajó en Google"),
)
def test_a_referent_to_something_never_delivered_is_answered_explicitly(message: str) -> None:
    """A falsified antecedent gets an explicit denial, not a guess or a clarification."""
    response = answer(message)

    assert response.trace.rendering_mode == "negative_referent"
    assert response.trace.grounding_status == "referent_missing"
    assert not response.trace.selected_fact_ids


@pytest.mark.parametrize("message", TERM_LEAKAGE_MESSAGES, ids=list(TERM_LEAKAGE_MESSAGES))
def test_query_terms_are_never_rendered_as_missing_entities(message: str) -> None:
    """`unmatched_terms` are query residue; a pronoun or verb is not a missing entity."""
    response = answer(message)
    normalized = response.answer.casefold()

    for term in ("his", "resume,", "marco."):
        assert term not in normalized


@pytest.mark.parametrize("message", TERM_LEAKAGE_MESSAGES, ids=list(TERM_LEAKAGE_MESSAGES))
def test_a_pronoun_does_not_make_a_stocked_topic_look_empty(message: str) -> None:
    """Both phrasings ask for the projects the profile does hold; answer them."""
    response = answer(message)

    assert response.trace.grounding_status != "profile_missing"
    assert response.trace.selected_fact_ids


@pytest.mark.parametrize(
    "message",
    ("Did Marco work at Google?", "Tell me about Marco's experience at Google."),
)
def test_the_same_absent_entity_resolves_alike_across_phrasings(message: str) -> None:
    """Sentence length changed the verdict for Google; only the entity may decide it."""
    response = answer(message)

    assert response.trace.grounding_status == "profile_missing"
    assert "Google" in response.answer


# --- #19 second half: the record decides whether a referent was ever delivered ---


def turn(message: str, state: object | None = None) -> object:
    """One offline turn that can be handed the state the previous one produced."""
    service = AgentService(
        profile=load_profile("data/profile.json"),
        classifier=StubClassifier(),
        generator=EmptyFactSetIsAFailure(),
    )
    return service.respond(message, history=[], state=state)


GLOBAL_PAYMENTS_REFERENT = "The part where it says he worked at Global Payments"


def test_a_referent_to_undelivered_profile_content_is_corrected_then_answered() -> None:
    """The premise is false but the content is real: say so, then answer it."""
    first = turn("Tell me about Sybil")
    assert first.state is not None and first.state.delivered_fact_ids

    second = turn(GLOBAL_PAYMENTS_REFERENT, first.state)

    assert second.trace.referent_correction is True
    assert second.trace.selected_fact_ids
    assert "Global Payments" in second.answer
    # The answer below the correction is whatever the ordinary path would have given.
    assert second.trace.rendering_mode in {"canonical", "rephrased", "transformed"}


def test_a_referent_to_content_already_delivered_is_answered_normally() -> None:
    """Once the record shows it was said, the referent is genuine and needs no denial."""
    first = turn("Dime acerca de la experiencia de Marco")
    assert first.state is not None and first.state.delivered_fact_ids

    second = turn(GLOBAL_PAYMENTS_REFERENT, first.state)

    assert second.trace.referent_correction is False
    assert second.trace.rendering_mode != "negative_referent"


def test_the_record_is_what_separates_the_two_verdicts() -> None:
    """Same message, same profile: only what was delivered may change the answer."""
    delivered = turn("Dime acerca de la experiencia de Marco").state
    unrelated = turn("Tell me about Sybil").state

    assert turn(GLOBAL_PAYMENTS_REFERENT, delivered).trace.referent_correction is False
    assert turn(GLOBAL_PAYMENTS_REFERENT, unrelated).trace.referent_correction is True


def test_an_antecedent_absent_from_the_profile_is_still_denied_outright() -> None:
    """The stronger claim survives: never said, and never in the profile either."""
    first = turn("Tell me about Sybil")

    second = turn("The part where it says he worked at google", first.state)

    assert second.trace.rendering_mode == "negative_referent"
    assert not second.trace.selected_fact_ids
