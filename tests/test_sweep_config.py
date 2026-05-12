from __future__ import annotations

from argparse import Namespace

from run_top_aware_muon_sweep import STRUCTURED_REFERENCE_CONFIGS, structured_config


def _args(mode: str) -> Namespace:
    return Namespace(
        structured_config=mode,
        precondition_frequency=5,
        shampoo_beta=0.5,
        optimizer_beta1=0.6,
        optimizer_beta2=0.7,
        structured_init_factor=2.0,
    )


def test_structured_config_auto_uses_method_reference_values() -> None:
    args = _args("auto")
    assert structured_config(args, "soap") == STRUCTURED_REFERENCE_CONFIGS["soap"]
    assert structured_config(args, "kl_soap") == STRUCTURED_REFERENCE_CONFIGS["kl_soap"]


def test_structured_config_global_uses_explicit_flags() -> None:
    cfg = structured_config(_args("global"), "soap")
    assert cfg == {
        "precondition_frequency": 5,
        "shampoo_beta": 0.5,
        "optimizer_beta1": 0.6,
        "optimizer_beta2": 0.7,
        "structured_init_factor": 2.0,
    }
