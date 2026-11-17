# ruff: noqa: EXE002
"""Tests for the response-blind E32.3 V2 input-generation repair."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import experiments.e32_interpbench.e32_3_v2_input_repair as repair
from experiments.e32_interpbench.adapter import OfficialDataPair


@dataclass(frozen=True)
class _Dataset:
    inputs: torch.Tensor
    targets: torch.Tensor


class _Case:
    def __init__(self, clean: _Dataset, corrupt: _Dataset) -> None:
        self.clean = clean
        self.corrupt = corrupt

    @staticmethod
    def is_categorical() -> bool:
        return True

    def get_clean_data(self, **_kwargs: object) -> _Dataset:
        return self.clean

    def get_corrupted_data(self, **_kwargs: object) -> _Dataset:
        return self.corrupt


class _AlwaysCorrectModel:
    def __init__(self, classes: int) -> None:
        self.classes = classes
        self.calls: list[torch.Tensor] = []

    def to(self, _device: str) -> _AlwaysCorrectModel:
        return self

    def eval(self) -> _AlwaysCorrectModel:
        return self

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls.append(inputs.detach().cpu().clone())
        output = torch.zeros(*inputs.shape, self.classes)
        output[..., 0] = 1
        return output


def _targets(count: int, positions: int = 2, classes: int = 2) -> torch.Tensor:
    value = torch.zeros(count, positions, classes)
    value[..., 0] = 1
    return value


def _pair(
    clean_rows: list[list[int]], source_rows: list[list[int]]
) -> OfficialDataPair:
    count = len(clean_rows)
    assert len(source_rows) == count
    return OfficialDataPair(
        clean_inputs=torch.tensor(clean_rows, dtype=torch.long),
        clean_targets=_targets(count),
        corrupt_inputs=torch.tensor(source_rows, dtype=torch.long),
        corrupt_targets=_targets(count),
    )


def test_duplicate_tolerant_generator_preserves_raw_order_and_pairing() -> None:
    clean = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    source = torch.tensor([[10, 11], [12, 13], [10, 11], [14, 15]])
    case = _Case(_Dataset(clean, _targets(4)), _Dataset(source, _targets(4)))

    pair = repair.generate_official_pair_allow_raw_duplicates(
        case, sample_count=4, clean_seed=1, corrupt_seed=2
    )
    audit = repair.raw_candidate_pool_audit(
        case_id=25,
        split="confirmation",
        candidate_count=4,
        clean_seed=1,
        corrupt_seed=2,
        pair=pair,
    )

    assert torch.equal(pair.clean_inputs, clean)
    assert torch.equal(pair.corrupt_inputs, source)
    assert audit["clean_unique_count"] == 4
    assert audit["corrupt_unique_count"] == 3
    assert audit["corrupt_duplicate_indices"] == ((2, 0),)
    assert audit["ordered_pair_count"] == 4


def test_duplicate_tolerant_selector_keeps_original_greedy_policy() -> None:
    candidates = _pair(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        [[20, 21], [2, 3], [22, 23], [22, 23], [24, 25], [26, 27]],
    )
    model = _AlwaysCorrectModel(classes=2)

    selected = repair._select_behaviorally_eligible_from_pair(
        model,
        _Case(
            _Dataset(candidates.clean_inputs, candidates.clean_targets),
            _Dataset(candidates.corrupt_inputs, candidates.corrupt_targets),
        ),
        candidates=candidates,
        required_count=3,
        candidate_count=6,
        device="cpu",
        excluded_token_rows=((0, 1),),
        require_unique_selected_token_rows=True,
    )

    assert selected.selected_candidate_indices == (2, 4, 5)
    assert selected.row_overlap_excluded_contexts == 3
    assert selected.row_disjoint_eligible_candidate_count == 3
    assert selected.selection_policy == (
        "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
        "response_blind_exclusions"
    )
    assert selected.eligibility_physical_forwards == 2
    assert selected.eligibility_model_examples == 12
    assert len(model.calls) == 2
    assert len(set(map(tuple, selected.pair.clean_inputs.tolist()))) == 3
    assert len(set(map(tuple, selected.pair.corrupt_inputs.tolist()))) == 3


def test_production_duplicate_path_rehashes_candidates_before_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _AlwaysCorrectModel(classes=2)
    drifted = _pair([[0, 1]], [[2, 3]])
    monkeypatch.setattr(
        repair,
        "generate_official_pair_allow_raw_duplicates",
        lambda *_args, **_kwargs: drifted,
    )

    with pytest.raises(ValueError, match="raw candidate identity count changed"):
        repair.behaviorally_eligible_pair_allow_raw_duplicates(
            model,
            _Case(
                _Dataset(drifted.clean_inputs, drifted.clean_targets),
                _Dataset(drifted.corrupt_inputs, drifted.corrupt_targets),
            ),
            required_count=128,
            candidate_count=192,
            clean_seed=725_200,
            corrupt_seed=725_201,
            device="cpu",
            excluded_token_rows=(),
            require_unique_selected_token_rows=True,
        )
    assert model.calls == []


def test_response_blind_capacity_bound_is_conservative() -> None:
    smoke_rows = ((90, 91),)
    discovery = _pair(
        [[0, 1], [2, 3], [4, 5]],
        [[10, 11], [12, 13], [14, 15]],
    )
    confirmation = _pair(
        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
        [[30, 31], [0, 1], [32, 33], [34, 35], [22, 23]],
    )

    audit = repair.response_blind_capacity_bound(
        smoke_rows=smoke_rows,
        discovery_pair=discovery,
        confirmation_pair=confirmation,
    )

    assert audit["discovery_individually_fresh_candidate_count"] == 3
    assert audit["confirmation_capacity_after_smoke"] == 4
    assert audit["confirmation_indices_touching_any_discovery_candidate"] == (1,)
    assert audit["confirmation_capacity_lower_bound_after_any_discovery"] == 3


def test_v1_failure_state_and_hash_chain_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = repair.verify_v1_failure_state()

    assert audit["failure_stage"] == "global_behavior_screen_case25_confirmation"
    assert audit["patched_intervention_model_examples"] == 0
    assert audit["v1_behavior_only_model_examples"] == 1_344
    assert audit["v1_behavior_only_physical_forwards"] == 10

    monkeypatch.setattr(
        repair,
        "V1_ARTIFACT_SHA256",
        {**repair.V1_ARTIFACT_SHA256, "failure": "0" * 64},
    )
    with pytest.raises(ValueError, match="failure.*SHA-256"):
        repair.verify_v1_failure_state()


def test_recovery_addendum_binds_only_the_operational_amendment() -> None:
    addendum = repair.recovery_addendum(
        v1_state={"verified": True},
        input_preflight={"verified": True},
    )

    assert addendum["parent_v1_protocol_sha256"] == repair.V1_PROTOCOL_SHA256
    assert addendum["parent_v1_split_alignment_sha256"] == (
        repair.V1_SPLIT_ALIGNMENT_SHA256
    )
    assert addendum["sole_amendment"] == {
        "case_id": 25,
        "split": "confirmation",
        "candidate_count": 192,
        "preserve_original_order_pairing_targets_and_seeds": True,
        "bypass_only_full_raw_role_uniqueness_assertion": True,
        "deduplicate_reorder_oversample_reseed_retry": False,
        "selected_row_uniqueness_and_behavioral_gate_unchanged": True,
    }
    assert addendum["v2_planned_query_accounting"]["model_examples"] == 302_784
    assert addendum["cumulative_if_v2_succeeds"]["model_examples"] == 304_128
    assert addendum["scientific_protocol_changed"] is False


def test_registration_dependencies_bind_v2_source_parent_and_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair._V1ProductionExecutor,
        "registration_dependencies",
        lambda _self: {"frozen_v1_dependencies": True},
    )
    monkeypatch.setattr(
        repair,
        "verify_v1_failure_state",
        lambda: {"verified": True, "parent": "v1"},
    )
    monkeypatch.setattr(
        repair,
        "_preflight_all_input_pools",
        lambda: ({}, {"verified": True, "model_calls": 0}),
    )

    dependencies = repair._RecoveryExecutor().registration_dependencies()

    assert dependencies["frozen_v1_dependencies"] is True
    recovery = dependencies["e32_3_v2_input_repair"]
    assert recovery["parent_v1_terminal_state"] == {
        "verified": True,
        "parent": "v1",
    }
    assert recovery["response_blind_input_preflight"] == {
        "verified": True,
        "model_calls": 0,
    }
    assert recovery["v2_source_sha256"] == repair.sha256_file(repair.__file__)


def test_screen_preflight_completes_before_any_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cases = {case_id: object() for case_id in repair.CASE_ORDER}

    def preflight() -> tuple[dict[int, object], dict[str, object]]:
        events.append("preflight")
        return cases, {"verified": True}

    def load_model(case_id: int, *, device: str) -> object:
        assert device == "cpu"
        events.append(f"model:{case_id}")
        return object()

    def acquire(case_id: int, **_kwargs: object) -> object:
        events.append(f"panel:{case_id}")
        return SimpleNamespace(case_id=case_id)

    monkeypatch.setattr(repair, "_preflight_all_input_pools", preflight)
    monkeypatch.setattr(repair, "load_verified_model", load_model)
    monkeypatch.setattr(repair, "acquire_case_panel", acquire)
    monkeypatch.setattr(repair, "_acquire_case25_panel_v2", acquire)
    monkeypatch.setattr(
        repair,
        "_behavior_screen_record",
        lambda panel: {"case_id": panel.case_id},
    )

    records, panels = repair._RecoveryExecutor().screen_all_cases()

    assert events[0] == "preflight"
    assert events[1:] == [
        "model:58",
        "panel:58",
        "model:129",
        "panel:129",
        "model:25",
        "panel:25",
    ]
    assert tuple(panels) == repair.CASE_ORDER
    assert tuple(records) == tuple(map(str, repair.CASE_ORDER))
    assert records["25"]["e32_3_v2_input_repair"]["verified"] is True


def test_case25_panel_changes_only_confirmation_raw_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = ((90, 91), (92, 93))
    discovery_pair = _pair(
        [[1_000 + 2 * index, 1_001 + 2 * index] for index in range(64)],
        [[2_000 + 2 * index, 2_001 + 2 * index] for index in range(64)],
    )
    confirmation_pair = _pair(
        [[3_000 + 2 * index, 3_001 + 2 * index] for index in range(128)],
        [[4_000 + 2 * index, 4_001 + 2 * index] for index in range(128)],
    )
    calls: list[tuple[str, int, int, int]] = []

    monkeypatch.setattr(repair, "smoke_excluded_rows", lambda *_args: smoke)

    def discovery_select(
        _model: object,
        _case: object,
        *,
        required_count: int,
        candidate_count: int,
        excluded_token_rows: tuple[tuple[int, ...], ...],
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append(
            (
                "strict_discovery",
                required_count,
                candidate_count,
                len(excluded_token_rows),
            )
        )
        return SimpleNamespace(pair=discovery_pair)

    def confirmation_select(
        _model: object,
        _case: object,
        *,
        required_count: int,
        candidate_count: int,
        excluded_token_rows: tuple[tuple[int, ...], ...],
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append(
            (
                "duplicate_tolerant_confirmation",
                required_count,
                candidate_count,
                len(excluded_token_rows),
            )
        )
        return SimpleNamespace(pair=confirmation_pair)

    monkeypatch.setattr(repair, "behaviorally_eligible_pair", discovery_select)
    monkeypatch.setattr(
        repair,
        "behaviorally_eligible_pair_allow_raw_duplicates",
        confirmation_select,
    )
    monkeypatch.setattr(repair, "validate_case_panel", lambda _panel: {"passed": True})

    panel = repair._acquire_case25_panel_v2(
        25, model=object(), case=object(), device="cpu"
    )

    assert panel.discovery.pair is discovery_pair
    assert panel.confirmation.pair is confirmation_pair
    assert calls == [
        ("strict_discovery", 64, 96, 2),
        ("duplicate_tolerant_confirmation", 128, 192, 130),
    ]


def test_v2_paths_accounting_and_public_entrypoints_are_frozen() -> None:
    assert repair.CANONICAL_REGISTRATION_PATH.name == (
        "E32.3-SPARSE-CORE-V2-INPUT-REPAIR-REGISTRATION.json"
    )
    assert repair.CANONICAL_RUN_ATTEMPT_PATH.name == (
        "E32.3-SPARSE-CORE-V2-INPUT-REPAIR-RUN-ATTEMPT.json"
    )
    assert repair.CANONICAL_RUN_DIRECTORY.name == ("e32_3_sparse_core_v2_input_repair")
    assert tuple(inspect.signature(repair.seal_registration).parameters) == ()
    assert tuple(inspect.signature(repair.build_registration_record).parameters) == ()
    assert tuple(inspect.signature(repair.run_registered_experiment).parameters) == (
        "confirm",
    )
    assert (
        inspect.signature(repair.run_registered_experiment).parameters["confirm"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_score_overlay_discloses_behavioral_reuse_without_changing_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair._V1ProductionExecutor,
        "score_complete_panel",
        lambda _self, _panel: {"overall_sparse_core_generalization": False},
    )

    score = repair._RecoveryExecutor().score_complete_panel({})

    assert score["overall_sparse_core_generalization"] is False
    disclosure = score["e32_3_v2_operational_recovery"]
    assert disclosure["v1_behavioral_endpoints_partially_opened"] is True
    assert disclosure["v1_patched_intervention_outcomes_opened"] is False
    assert disclosure["v2_patched_intervention_outcomes_prospective"] is True
    assert disclosure["ordinary_end_to_end_pre_behavior_preregistration"] is False
