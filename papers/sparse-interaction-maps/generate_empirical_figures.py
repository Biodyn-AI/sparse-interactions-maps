#!/usr/bin/env python3
"""Render vector figures from audited scalar records; never run a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
MANIFEST_PATH = PAPER / "evidence" / "scalar_manifest.json"
FRESH_PATH = ROOT / "runs/e33_sparse_interaction_diagnostics_v1/summary.json"
FIGURES = PAPER / "figures"
COLORS = {"direct": "#236B8E", "transitive_only": "#DBA740", "unrelated": "#9DA7AB"}
LABELS = {"direct": "Direct link", "transitive_only": "Linked through more steps",
          "unrelated": "No path in known task circuit"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_and_collect(manifest: dict, fresh: dict) -> dict:
    for key in ("e31_2", "e32_1", "e32_1_confirmation", "e32_2"):
        source = manifest["sources"][key]
        assert sha256(Path(source["absolute_path"])) == source["sha256"], key
    score = manifest["e31_2"]["score"]
    assert score["primary_verdict"].lower() == "fail"
    assert (score["primary"]["selected_edge_count"], score["primary"]["total_edge_count"]) == (48, 432)
    parents = ("l05a_l08a", "l08a_l09a", "l05a_l10a")
    heads = []
    for parent in parents:
        heads.append({
            "parent": parent,
            "confirmation_energy_coverage": score["parents"][parent]["selected_confirmation_energy_coverage"],
            "pair_group_r2": score["reconstruction"]["pair_selected_composite"]["parents"][parent]["r2"],
            "singleton_group_r2": score["reconstruction"]["singleton_selected_composite"]["parents"][parent]["r2"],
        })
    posthoc = manifest["e32_1"]["posthoc_reachability"]
    assert (posthoc["top10_direct_count"], posthoc["top10_transitive_nonedge_count"],
            posthoc["top10_unrelated_count"]) == (5, 5, 0)
    fresh_cases = {case["case_id"]: case for case in fresh["cases"]}
    assert set(fresh_cases) == {63, 103, 82}
    fresh_rows = []
    for case_id in (63, 103, 82):
        case = fresh_cases[case_id]
        classes = case["class_statistics"]
        assert abs(sum(item["mass_fraction"] for item in classes.values()) - 1.0) < 1e-12
        assert abs(sum(item["class_fraction"] for item in classes.values()) - 1.0) < 1e-12
        assert sum(item["count"] for item in classes.values()) == case["candidate_pair_count"]
        fresh_rows.append({
            "case_id": case_id, "task": case["task"],
            "candidate_pair_count": case["candidate_pair_count"],
            "classes": classes,
            "discovery_confirmation_spearman": case["discovery_confirmation_spearman"],
            "rank_one_squared_score_norm_explained": case["rank_one_endpoint_factor"]["explained_squared_score_norm"],
            "direct_auprc": case["ranking"]["direct"]["auprc"],
            "closure_auprc": case["ranking"]["closure"]["auprc"],
        })
    # These comparisons guard against accidentally swapping observed mass and
    # reference prevalence, which have very different interpretations here.
    assert fresh_rows[1]["classes"]["unrelated"]["mass_fraction"] > 0.54
    assert fresh_rows[1]["classes"]["unrelated"]["class_fraction"] > 0.93
    assert fresh_rows[2]["classes"]["unrelated"]["mass_fraction"] > 0.43
    table = []
    for case_id in (2, 21):
        case = manifest["e32_2"]["case_results"][str(case_id)]
        methods = case["confirmation_metrics"]
        best = case["best_eap_variant"]
        table.append({
            "case_id": case_id, "raw_pair_ap": methods["raw_pair_interaction"]["auprc"],
            "node_factor_ap": methods["rank_one_node_factor"]["auprc"],
            "directed_replacement_ap": methods["bidirectional_fine_replacement"]["auprc"],
            "best_eap_variant": best, "best_eap_ap": methods[best]["auprc"],
            "candidate_fine_edge_fraction": case["structure"]["induced_fine_edge_fraction"],
            "registered_useful_pass": case["gates"]["useful_pass"],
        })
    return {
        "schema_version": "sparse_interaction_paper_figure_data_v1",
        "sources": {"scalar_manifest": {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH)},
                    "e33_summary": {"path": str(FRESH_PATH), "sha256": sha256(FRESH_PATH)}},
        "head_parents": heads, "case97_posthoc": posthoc,
        "fresh_cases": fresh_rows, "e32_2_table_rows": table,
        "model_calls": 0,
    }


def style() -> None:
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def historical_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45), gridspec_kw={"width_ratios": [1, 1.18, 1]})
    x = np.arange(3)
    parents = data["head_parents"]
    cover = [row["confirmation_energy_coverage"] for row in parents]
    axes[0].bar(x, cover, color=COLORS["direct"], width=0.56)
    axes[0].set(title="(a) Interaction strength captured", ylabel="Share on new prompts", ylim=(0, 1.12))
    axes[0].set_xticks(x, ["L5 x L8", "L8 x L9", "L5 x L10"])
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for i, value in enumerate(cover):
        axes[0].text(i, value + 0.025, f"{100*value:.1f}%", ha="center", fontsize=10)
    axes[0].text(0.5, -0.30, "16 of 144 pairs per parent;\n48 of 432 pairs overall", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=10)
    pair = [row["pair_group_r2"] for row in parents]
    single = [row["singleton_group_r2"] for row in parents]
    axes[1].bar(x - 0.17, pair, width=0.32, color=COLORS["direct"], label="Chosen from pair tests")
    axes[1].bar(x + 0.17, single, width=0.32, color="#75A992", label="Chosen from single-part tests")
    axes[1].set(title="(b) Reproducing the interaction", ylabel="Agreement ($R^2$)", ylim=(0, 1.12))
    axes[1].set_xticks(x, ["L5 x L8", "L8 x L9", "L5 x L10"])
    axes[1].legend(loc="lower left", bbox_to_anchor=(-0.03, -0.41), frameon=False, ncol=1)
    categories = ("direct", "transitive_only", "unrelated")
    record = data["case97_posthoc"]
    values = [record["direct_energy_fraction"], record["transitive_nonedge_energy_fraction"], record["unrelated_energy_fraction"]]
    axes[2].bar(x, values, color=[COLORS[key] for key in categories], width=0.58)
    axes[2].set(title="(c) Case 97: where scores go", ylabel="Share of total pair score", ylim=(0, 0.70))
    axes[2].set_xticks(x, ["Direct\nlink", "Longer\npath", "No known\npath"])
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    for i, value in enumerate(values):
        axes[2].text(i, value + 0.025, f"{100*value:.1f}%", ha="center", fontsize=10)
    axes[2].text(0.5, -0.30, "Top ten: 5 direct links + 5 longer paths\nPath check added after the main test", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=10)
    fig.subplots_adjust(wspace=0.49, bottom=0.31, top=0.86)
    save(fig, "historical_diagnostics")


def fresh_figure(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    class_order = ("direct", "transitive_only", "unrelated")
    centers = np.arange(3) * 2.1
    width = 0.50
    positions = np.concatenate([[center - 0.29, center + 0.29] for center in centers])
    bottoms = np.zeros(6)
    for key in class_order:
        values = []
        for row in data["fresh_cases"]:
            values.extend([row["classes"][key]["mass_fraction"], row["classes"][key]["class_fraction"]])
        bars = ax.bar(positions, values, bottom=bottoms, width=width,
                      color=COLORS[key], label=LABELS[key], edgecolor="white", linewidth=0.8)
        for bar, value, bottom in zip(bars, values, bottoms, strict=True):
            if value >= 0.075:
                ax.text(bar.get_x()+bar.get_width()/2, bottom+value/2, f"{100*value:.1f}%",
                        ha="center", va="center", color="white" if key == "direct" else "#17242B", fontsize=10)
        bottoms += values
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Fraction of total score or tested pairs")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(positions, ["Score\nshare", "Share of\npairs"] * 3)
    for center, row in zip(centers, data["fresh_cases"], strict=True):
        ax.text(center, -0.23, f"Case {row['case_id']} ({row['candidate_pair_count']} pairs)",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.set_title("New models: scores versus known task circuits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.32)
    save(fig, "fresh_replication")


def write_e32_2_table(data: dict) -> None:
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Case & Pair AP & Product AP & Link-change AP & Best EAP AP & Links tested \\",
            r"\midrule"]
    for row in data["e32_2_table_rows"]:
        rows.append(f"{row['case_id']} & {row['raw_pair_ap']:.3f} & {row['node_factor_ap']:.3f} & "
                    f"{row['directed_replacement_ap']:.3f} & {row['best_eap_ap']:.3f} & "
                    f"{100*row['candidate_fine_edge_fraction']:.1f}\\% " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             "% AP is on promoted whole-component labels. Candidate fraction uses fine Q/K/V edges.",
             "% Best EAP* variant is eap_absolute_max in case 2 and eap_ig_signed_max in case 21.",
             "% Both cases fail the registered useful and superiority conjunctions."]
    (FIGURES / "e32_2_localization_table.tex").write_text("\n".join(rows) + "\n")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    fresh = json.loads(FRESH_PATH.read_text())
    collected = verify_and_collect(manifest, fresh)
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    historical_figure(collected)
    fresh_figure(collected)
    write_e32_2_table(collected)
    (PAPER / "evidence" / "figure_source_data.json").write_text(
        json.dumps(collected, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("Verified plotted scalars and wrote two PDF/PNG figure pairs; no model calls.")


if __name__ == "__main__":
    main()
