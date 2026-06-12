"""Tests for serializing Tool 3's runtime results into the .cls.json ``divergences[]`` contract.

Pure-Python (no torch): map the runtime ``DivergenceFindings`` / ``CausalAttribution`` dataclasses
into the pydantic schema record and check it round-trips through the artifact.
"""

from compile_lens import _schema
from compile_lens.tools.divergence import (
    CausalAttribution,
    DivergenceFindings,
    divergence_to_record,
)


def test_record_maps_a_localized_finding_without_attribution() -> None:
    findings = DivergenceFindings(
        first_divergent_layer="decoder.layers.3",
        max_abs_diff=0.5,
        num_layers_compared=10,
        rtol=1e-3,
        atol=1e-5,
        suggested_cause=None,
    )
    record = divergence_to_record(findings, divergence_id="dv_1")

    assert isinstance(record, _schema.Divergence)
    assert record.divergence_id == "dv_1"
    assert record.first_divergent_layer == "decoder.layers.3"
    assert record.max_abs_diff == 0.5
    assert record.num_layers_compared == 10
    assert record.rtol == 1e-3
    assert record.suggested_cause is None
    assert record.attribution is None


def test_record_nests_the_causal_attribution() -> None:
    # A no-divergence finding (first_divergent_layer None) plus a structured attribution.
    findings = DivergenceFindings(None, None, 5, 1e-3, 1e-5, suggested_cause="cause string")
    attribution = CausalAttribution(
        responsible_passes=["epilogue_fusion"],
        attributed=True,
        summary="cause string",
        num_probes=4,
    )
    record = divergence_to_record(findings, divergence_id="dv_2", attribution=attribution)

    assert record.first_divergent_layer is None
    assert record.suggested_cause == "cause string"
    assert record.attribution is not None
    assert record.attribution.attributed is True
    assert record.attribution.responsible_passes == ["epilogue_fusion"]
    assert record.attribution.num_probes == 4


def test_record_round_trips_through_the_artifact() -> None:
    """findings -> record -> ClsArtifact -> to_json -> from_json preserves the typed section."""
    findings = DivergenceFindings("layer.7", 0.25, 12, 1e-3, 1e-5, suggested_cause="x")
    attribution = CausalAttribution(["pattern_matcher"], True, "x", 6)
    record = divergence_to_record(findings, divergence_id="dv_3", attribution=attribution)

    session = _schema.Session(
        id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-06-12T00:00:00Z",
        torch_version="2.6.0",
        redaction_policy=_schema.RedactionPolicy.DEFAULT_STRICT,
    )
    artifact = _schema.ClsArtifact(schema_version="0.5.0", session=session, divergences=[record])

    reparsed = _schema.from_json(_schema.to_json(artifact))

    assert len(reparsed.divergences) == 1
    assert reparsed.divergences[0].first_divergent_layer == "layer.7"
    assert reparsed.divergences[0].max_abs_diff == 0.25
    assert reparsed.divergences[0].attribution is not None
    assert reparsed.divergences[0].attribution.responsible_passes == ["pattern_matcher"]


def test_empty_divergences_is_omitted_on_serialize() -> None:
    # A non-Tool-3 artifact must not grow a spurious "divergences": [].
    session = _schema.Session(
        id="00000000-0000-4000-8000-000000000000",
        timestamp="2026-06-12T00:00:00Z",
        torch_version="2.6.0",
        redaction_policy=_schema.RedactionPolicy.DEFAULT_STRICT,
    )
    artifact = _schema.ClsArtifact(schema_version="0.5.0", session=session)
    assert "divergences" not in _schema.to_json(artifact)
