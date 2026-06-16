"""Non-finite float policy (ADR-039): NaN / ±inf are written as the explicit JSON string sentinels
``"NaN"`` / ``"Infinity"`` / ``"-Infinity"``, never the silent ``null`` both serializers default to.
This is the Python (writer) half of the policy; mirrors the Rust crate's ``tests/non_finite.rs``.
"""

import json
import math

import pytest
from pydantic import ValidationError

from compile_lens._schema import Divergence, KernelMeasurements, to_json


def test_non_finite_writes_sentinels_not_null() -> None:
    m = KernelMeasurements(mean_us=math.nan, median_us=math.inf, p99_us=-math.inf)
    dumped = m.model_dump_json(exclude_unset=True)
    out = json.loads(dumped)
    assert out["mean_us"] == "NaN"
    assert out["median_us"] == "Infinity"
    assert out["p99_us"] == "-Infinity"
    assert "null" not in dumped  # the whole point: non-finite must not silently become null


def test_sentinels_read_back_to_floats() -> None:
    m = KernelMeasurements.model_validate_json(
        '{"mean_us":"NaN","median_us":"Infinity","p99_us":"-Infinity"}'
    )
    assert math.isnan(m.mean_us)
    assert m.median_us == math.inf
    assert m.p99_us == -math.inf


def test_nan_round_trips_losslessly() -> None:
    m = KernelMeasurements(mean_us=math.nan)
    back = KernelMeasurements.model_validate_json(m.model_dump_json(exclude_unset=True))
    assert math.isnan(back.mean_us)


def test_finite_stays_a_number() -> None:
    out = json.loads(KernelMeasurements(mean_us=12.5).model_dump_json(exclude_unset=True))
    assert out["mean_us"] == 12.5
    assert isinstance(out["mean_us"], float)


def test_unknown_string_is_rejected() -> None:
    # A non-sentinel string is a hard error, not a silently coerced/dropped value.
    with pytest.raises(ValidationError):
        KernelMeasurements.model_validate_json('{"mean_us":"bogus"}')


def test_max_abs_diff_nan_is_the_headline_case() -> None:
    # The field this policy exists for: a NaN max_abs_diff means the compiled model produced NaN at
    # that layer — Tool 3's strongest signal, which must survive the write instead of becoming null.
    d = Divergence(
        divergence_id="d1", num_layers_compared=3, max_abs_diff=math.nan, rtol=1e-3, atol=1e-5
    )
    out = json.loads(d.model_dump_json(exclude_unset=True))
    assert out["max_abs_diff"] == "NaN"
    back = Divergence.model_validate_json(d.model_dump_json(exclude_unset=True))
    assert math.isnan(back.max_abs_diff)


def test_to_json_write_path_emits_sentinel() -> None:
    # The canonical writer (to_json => exclude_unset) must also emit the sentinel, not null.
    from compile_lens._schema import ClsArtifact

    artifact = ClsArtifact.model_validate(
        {
            "schema_version": "0.5.0",
            "session": {
                "id": "00000000-0000-4000-8000-000000000000",
                "timestamp": "2026-01-01T00:00:00Z",
                "torch_version": "2.6.0",
                "redaction_policy": "default-strict",
            },
            "divergences": [
                {
                    "divergence_id": "d1",
                    "num_layers_compared": 3,
                    "max_abs_diff": float("nan"),
                    "rtol": 1e-3,
                    "atol": 1e-5,
                }
            ],
        }
    )
    out = json.loads(to_json(artifact))
    assert out["divergences"][0]["max_abs_diff"] == "NaN"
