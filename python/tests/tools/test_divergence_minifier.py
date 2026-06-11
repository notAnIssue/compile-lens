"""Tests for the accuracy-minifier integration (Tool 3): configure dynamo's accuracy minifier for
the duration of a block, then restore."""

import pytest

torch = pytest.importorskip("torch")

from compile_lens import accuracy_minifier  # noqa: E402
from compile_lens.tools.divergence import MinifierStatus  # noqa: E402


def test_sets_accuracy_minifier_config_and_restores() -> None:
    from torch._dynamo import config as dynamo_config

    before_level = dynamo_config.repro_level
    before_after = dynamo_config.repro_after

    with accuracy_minifier() as status:
        assert status.available
        # Inside the block the accuracy minifier is configured.
        assert dynamo_config.repro_level == 4
        assert dynamo_config.repro_after == "aot"

    # And the prior config is restored on exit, whatever it was.
    assert dynamo_config.repro_level == before_level
    assert dynamo_config.repro_after == before_after


def test_restores_even_on_exception() -> None:
    from torch._dynamo import config as dynamo_config

    before_level = dynamo_config.repro_level
    with pytest.raises(ValueError, match="boom"):
        with accuracy_minifier():
            assert dynamo_config.repro_level == 4
            raise ValueError("boom")
    assert dynamo_config.repro_level == before_level


def test_status_dataclass() -> None:
    ok = MinifierStatus(available=True)
    assert ok.available and ok.reason is None
    bad = MinifierStatus(available=False, reason="no repro config")
    assert not bad.available and bad.reason == "no repro config"


def test_composes_with_divergence_session() -> None:
    from compile_lens import divergence_session

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    other = torch.nn.Sequential(torch.nn.Linear(4, 4))
    x = torch.randn(2, 4)
    with accuracy_minifier() as m, divergence_session(model, other) as div:
        assert m.available
        model(x)
        other(x)
    # The session still captures and localizes with the minifier configured around it.
    assert div.captured_modules() == ["0"]
