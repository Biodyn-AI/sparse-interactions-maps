#!/usr/bin/env python3
"""Export publication-size vector figures from immutable audited values.

No model is loaded, no scientific source is rewritten, and no PLOS or other
Scientific Reports submission is touched. The PDF canvas is exactly 7 inches
wide. Optional TIFF copies are 600 dpi, RGB, and LZW compressed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
SOURCE = PAPER / "evidence/figure_source_data.json"
SOURCE_SHA256 = "f4f104fdf2583e139c92ea32d233ee631621078afc9ff3f1ceb61a6cc54d77a2"
OUTPUT = ROOT / "output/scientific-reports-sparse-interactions/upload"
COLORS = {"direct": "#236B8E", "transitive_only": "#DBA740", "unrelated": "#9DA7AB"}
SINGLE_COLOR = "#75A992"
CLASS_ORDER = ("direct", "transitive_only", "unrelated")


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_source() -> dict:
    """Fail on source drift before plotting; never trust rounded manuscript text."""
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Audited figure-source bytes changed; review before exporting.")
    source = json.loads(SOURCE.read_text())
    if source["schema_version"] != "sparse_interaction_paper_figure_data_v1" or source["model_calls"] != 0:
        raise ValueError("Unexpected figure-source schema or acquisition status.")
    if [row["parent"] for row in source["head_parents"]] != ["l05a_l08a", "l08a_l09a", "l05a_l10a"]:
        raise ValueError("Head-pair order changed.")
    if [row["case_id"] for row in source["fresh_cases"]] != [63, 103, 82]:
        raise ValueError("Fresh-case order changed.")
    for row in source["fresh_cases"]:
        for statistic in ("mass_fraction", "class_fraction"):
            values = [row["classes"][key][statistic] for key in CLASS_ORDER]
            if not all(0 <= value <= 1 for value in values) or not np.isclose(sum(values), 1, rtol=0, atol=1e-12):
                raise ValueError(f"Invalid class partition for case {row['case_id']}.")
    # Resolve actual source files locally, not through historical machine paths.
    sources = {"scalar_manifest": PAPER / "evidence/scalar_manifest.json",
               "e33_summary": ROOT / "runs/e33_sparse_interaction_diagnostics_v1/summary.json"}
    for name, path in sources.items():
        if sha256(path) != source["sources"][name]["sha256"]:
            raise ValueError(f"Underlying source hash changed: {name}")
    return source


def configure_style() -> str:
    arial = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if arial.is_file():
        font_manager.fontManager.addfont(str(arial))
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        family = "Arial"
    except ValueError:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 9,
                         "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.linewidth": 1.1,
                         "xtick.major.width": 1.1, "ytick.major.width": 1.1,
                         "xtick.minor.width": 1.1, "ytick.minor.width": 1.1,
                         "patch.linewidth": 1.1})
    return family


def panel_label(axis, letter: str) -> None:
    axis.text(-0.15, 1.19, letter, transform=axis.transAxes,
              fontsize=9, fontweight="bold", ha="left", va="bottom")


def check_bars(bars, expected, color: str) -> None:
    """Check rendered artists, not merely the inputs passed to Matplotlib."""
    if len(bars) != len(expected):
        raise ValueError("A plotted score is missing.")
    for bar, value in zip(bars, expected, strict=True):
        # Stacked rectangles can incur one subtraction-roundoff when their
        # height is reconstructed from the stored bottom and top coordinates.
        if not np.isclose(bar.get_height(), value, rtol=1e-14, atol=1e-15) or not np.allclose(bar.get_facecolor(), to_rgba(color), rtol=0, atol=1e-15):
            raise ValueError("A plotted score or class color differs from its source.")


def check_text_bounds(figure, number: int) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    for artist in figure.findobj(match=Text):
        if artist.get_visible() and artist.get_text():
            bounds = artist.get_window_extent(renderer)
            if (bounds.x0 < canvas.x0 - 1 or bounds.y0 < canvas.y0 - 1
                    or bounds.x1 > canvas.x1 + 1 or bounds.y1 > canvas.y1 + 1):
                raise ValueError(f"Figure {number} clips text: {artist.get_text()!r}")


def save_figure(figure, number: int, output: Path, family: str, tiff: bool) -> dict:
    check_text_bounds(figure, number)
    width, height = figure.get_size_inches()
    if not np.isclose(width, 7.0, rtol=0, atol=1e-12):
        raise ValueError("Publication width must be exactly 7 inches.")
    pdf = output / f"Fig{number}.pdf"
    figure.savefig(pdf, metadata={"Creator": "Matplotlib: audited saved scientific values",
                                 "CreationDate": None, "ModDate": None})
    record = {"file": pdf.name, "format": "native vector PDF", "width_inches": float(width),
              "height_inches": float(height), "size_bytes": pdf.stat().st_size,
              "sha256": sha256(pdf), "font": family, "font_points": [8, 9],
              "text_bounds_pass": True, "model_calls": 0}
    if tiff:
        path = output / f"Fig{number}.tif"
        figure.savefig(path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        with Image.open(path) as original:
            original.convert("RGB").save(path, compression="tiff_lzw", dpi=(600, 600))
        with Image.open(path) as checked:
            if checked.mode != "RGB" or checked.n_frames != 1 or checked.width != 4200:
                raise ValueError(f"Invalid publication TIFF: {path.name}")
            record["tiff"] = {"file": path.name, "pixels": list(checked.size), "dpi": 600,
                              "mode": checked.mode, "compression": checked.info.get("compression"),
                              "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    plt.close(figure)
    return record


def example_figure():
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.1))
    size = 8
    allowed = np.triu(np.ones((size, size), dtype=bool), k=1)
    direct = np.diag(np.ones(size - 1), k=1)
    weights = np.array([4, 2, 1, 0, 0, 0, 0, 0], dtype=float)
    interaction = 4 * np.outer(weights**2, weights**2)
    interaction /= interaction[allowed].max()
    titles = ("Copying chain:\nactual links", "Same chain:\ninteraction strength", "Shared output:\ninteraction strength")
    for axis, values, title, letter in zip(axes, (direct, np.ones((size, size)), interaction), titles, "abc", strict=True):
        palette = plt.get_cmap("Blues").with_extremes(bad="#e8e8e8")
        # Draw cells as vector polygons, rather than embedding a raster heatmap.
        axis.set_facecolor("#e8e8e8")
        axis.pcolormesh(np.arange(size + 1) - 0.5, np.arange(size + 1) - 0.5,
                        np.ma.masked_where(~allowed, values), cmap=palette,
                        vmin=0, vmax=1, shading="flat", rasterized=False)
        axis.set(xlim=(-0.5, size - 0.5), ylim=(size - 0.5, -0.5),
                 title=title, xlabel="Later part (known order)", ylabel="Earlier part")
        axis.set_aspect("equal")
        axis.set_xticks(range(size), range(1, size + 1))
        axis.set_yticks(range(size), range(1, size + 1))
        axis.tick_params(length=0)
        panel_label(axis, letter)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.76, wspace=0.38)
    return figure


def historical_figure(source: dict):
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.35), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    positions = np.arange(3)
    rows = source["head_parents"]
    shares = [row["confirmation_energy_coverage"] for row in rows]
    bars = axes[0].bar(positions, shares, color=COLORS["direct"], width=0.55)
    check_bars(bars, shares, COLORS["direct"])
    axes[0].set(title="Interaction strength\ncaptured", ylabel="Share on new prompts", ylim=(0, 1.15))
    axes[0].set_xticks(positions, ["L5 x L8", "L8 x L9", "L5 x L10"], rotation=20)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for index, value in enumerate(shares):
        axes[0].text(index, value + 0.03, f"{value:.1%}", ha="center", fontsize=8)
    axes[0].text(0.5, -0.35, "16 of 144 pairs per block pair;\n48 of 432 pairs overall",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=8)
    pair = [row["pair_group_r2"] for row in rows]
    single = [row["singleton_group_r2"] for row in rows]
    bars = axes[1].bar(positions - 0.17, pair, width=0.32, color=COLORS["direct"], label="Chosen from pair tests")
    check_bars(bars, pair, COLORS["direct"])
    bars = axes[1].bar(positions + 0.17, single, width=0.32, color=SINGLE_COLOR, label="Chosen from single-part tests")
    check_bars(bars, single, SINGLE_COLOR)
    axes[1].set(title="Reproducing\nthe interaction", ylabel="Agreement (R-squared)", ylim=(0, 1.15))
    axes[1].set_xticks(positions, ["L5 x L8", "L8 x L9", "L5 x L10"], rotation=20)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    record = source["case97_posthoc"]
    shares = [record[key] for key in ("direct_energy_fraction", "transitive_nonedge_energy_fraction", "unrelated_energy_fraction")]
    for position, value, key in zip(positions, shares, CLASS_ORDER, strict=True):
        check_bars(axes[2].bar([position], [value], color=COLORS[key], width=0.58), [value], COLORS[key])
        axes[2].text(position, value + 0.02, f"{value:.1%}", ha="center", fontsize=8)
    axes[2].set(title="Case 97:\nwhere scores go", ylabel="Share of total pair score", ylim=(0, 0.7))
    axes[2].set_xticks(positions, ["Direct\nlink", "Longer\npath", "No\nknown\npath"])
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].text(0.5, -0.35, "Top ten: 5 direct links\n+ 5 longer paths\nPath check added\nafter the main test",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=8)
    for axis, letter in zip(axes, "abc", strict=True):
        panel_label(axis, letter)
    figure.subplots_adjust(left=0.10, right=0.97, wspace=0.80, top=0.77, bottom=0.37)
    return figure


def fresh_figure(source: dict):
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    centers = np.arange(3) * 2.1
    positions = np.concatenate([[center - 0.29, center + 0.29] for center in centers])
    bottoms = np.zeros(6)
    labels = ("Direct link", "Linked through more steps", "No path in known task circuit")
    for key, label in zip(CLASS_ORDER, labels, strict=True):
        heights = [row["classes"][key][statistic] for row in source["fresh_cases"]
                   for statistic in ("mass_fraction", "class_fraction")]
        bars = axis.bar(positions, heights, bottom=bottoms, width=0.5, color=COLORS[key],
                        label=label, edgecolor="white", linewidth=1.1)
        check_bars(bars, heights, COLORS[key])
        for bar, height, bottom in zip(bars, heights, bottoms, strict=True):
            if height >= 0.075:
                axis.text(bar.get_x() + bar.get_width() / 2, bottom + height / 2, f"{height:.1%}",
                          ha="center", va="center", fontsize=8,
                          color="white" if key == "direct" else "#17242B")
        bottoms += heights
    if not np.allclose(bottoms, np.ones(6), rtol=0, atol=1e-12):
        raise ValueError("Stacked bar totals differ from the source class partitions.")
    axis.set(ylim=(0, 1.06), ylabel="Fraction of total score\nor tested pairs")
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xticks(positions, ["Score\nshare", "Share of\npairs"] * 3)
    for center, row in zip(centers, source["fresh_cases"], strict=True):
        axis.text(center, -0.24, f"Case {row['case_id']} ({row['candidate_pair_count']} pairs)",
                  transform=axis.get_xaxis_transform(), ha="center", fontsize=9)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.36)
    return figure


def make_figures(output: Path = OUTPUT, tiff: bool = True) -> list[dict]:
    source = load_source()
    family = configure_style()
    output.mkdir(parents=True, exist_ok=True)
    figures = (example_figure(), historical_figure(source), fresh_figure(source))
    return [save_figure(figure, number, output, family, tiff)
            for number, figure in enumerate(figures, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pdf-only", action="store_true")
    options = parser.parse_args()
    print(json.dumps({"source_sha256": SOURCE_SHA256, "figures": make_figures(options.output, not options.pdf_only),
                      "model_calls": 0, "network_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
