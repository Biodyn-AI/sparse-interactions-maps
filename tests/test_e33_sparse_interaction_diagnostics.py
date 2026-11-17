"""Checks for the frozen, bounded E33 reachability diagnostic."""

import pytest
import torch

from experiments.e33_sparse_interaction_diagnostics.study import (
    class_statistics,
    protocol,
    transitive_closure,
    validate_source_hashes,
)


def test_closure_separates_direct_transitive_and_unrelated():
    direct = (("a", "b"), ("b", "c"))
    assert transitive_closure(direct) == {("a", "b"), ("b", "c"), ("a", "c")}
    edges = (("a", "b"), ("b", "c"), ("a", "c"), ("a", "d"))
    stats = class_statistics(torch.tensor([1.0, 1.0, 6.0, 2.0]), edges, direct)
    assert stats["direct"]["mass_fraction"] == pytest.approx(0.2)
    assert stats["transitive_only"]["mass_fraction"] == pytest.approx(0.6)
    assert stats["transitive_only"]["class_fraction"] == pytest.approx(0.25)
    assert stats["transitive_only"]["mass_enrichment"] == pytest.approx(2.4)
    assert stats["unrelated"]["mean_score"] == pytest.approx(2.0)


def test_no_path_control_reports_absent_class_without_fake_zero_evidence():
    stats = class_statistics(
        torch.tensor([3.0, 1.0]), (("a", "b"), ("a", "c")), (("a", "b"),)
    )
    assert stats["transitive_only"]["count"] == 0
    assert stats["transitive_only"]["mean_score"] is None
    assert stats["transitive_only"]["mass_enrichment"] is None


def test_cycles_are_rejected():
    with pytest.raises(ValueError, match="acyclic"):
        transitive_closure((("a", "b"), ("b", "a")))


def test_source_hash_drift_fails_closed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    import hashlib

    expected = {str(source): hashlib.sha256(b"original").hexdigest()}
    validate_source_hashes(expected)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        validate_source_hashes(expected)


def test_protocol_is_bounded_and_runs_every_case():
    frozen = protocol()
    assert frozen["case_order"] == [63, 103, 82]
    assert frozen["all_cases_run_regardless_of_outcomes"] is True
    assert frozen["maximum_model_examples"] == 85_440
    assert frozen["maximum_model_examples"] <= 90_000
    assert frozen["bootstrap_replicates"] == 1_000
