"""Generate scientific figures and compile the paper without executing models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
REPOSITORY = PAPER.parents[1]
OUTPUT = REPOSITORY / "output" / "pdf"
FIGURES = PAPER / "figures"
BUILD = PAPER / "build"


def plot_examples() -> None:
    """Plot exact examples; missing/invalid matrix cells stay explicitly gray."""
    width = 8
    valid = np.triu(np.ones((width, width), dtype=bool), k=1)
    direct = np.zeros((width, width))
    for index in range(width - 1):
        direct[index, index + 1] = 1
    weights = np.array([4, 2, 1, 0, 0, 0, 0, 0], dtype=float)
    quadratic = 4 * np.outer(weights**2, weights**2)
    quadratic /= quadratic[valid].max()
    panels = (direct, np.ones((width, width)), quadratic)
    titles = (
        "Copying chain:\nactual links", "Same chain:\ninteraction strength",
        "Shared output:\ninteraction strength",
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.45), constrained_layout=True)
    for axis, values, title in zip(axes, panels, titles, strict=True):
        color_map = plt.get_cmap("Blues").copy()
        color_map.set_bad("#e8e8e8")
        axis.imshow(
            np.ma.masked_where(~valid, values), cmap=color_map,
            vmin=0, vmax=1, interpolation="nearest",
        )
        axis.set_title(title, fontsize=11, pad=9)
        axis.set_xticks(range(width), range(1, width + 1))
        axis.set_yticks(range(width), range(1, width + 1))
        axis.set_xlabel("Later part (known order)")
        axis.set_ylabel("Earlier part")
        axis.tick_params(length=0)
    figure.savefig(FIGURES / "illustrative_counterexamples.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "illustrative_counterexamples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def compile_pdf() -> Path:
    for executable in ("pdflatex", "bibtex"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required LaTeX executable is missing: {executable}")
    environment = os.environ.copy()
    environment["BIBINPUTS"] = str(PAPER) + os.pathsep + environment.get("BIBINPUTS", "")
    tex_command = [
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        f"-output-directory={BUILD}", "manuscript.tex",
    ]
    commands = [tex_command, ["bibtex", "manuscript"], tex_command, tex_command]
    for index, command in enumerate(commands, start=1):
        completed = subprocess.run(
            command, cwd=BUILD if command[0] == "bibtex" else PAPER, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        (BUILD / f"compile-{index}.log").write_text(completed.stdout)
        if completed.returncode:
            print(completed.stdout[-7000:], file=sys.stderr)
            raise RuntimeError(f"LaTeX build step {index} failed")
    log_text = (BUILD / "manuscript.log").read_text()
    prohibited = ("undefined citations", "There were undefined references", "LaTeX Error:")
    if any(message in log_text for message in prohibited):
        raise RuntimeError("The final LaTeX log contains unresolved references or errors")
    output = OUTPUT / "sparse-interaction-maps-draft.pdf"
    shutil.copy2(BUILD / "manuscript.pdf", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    for directory in (FIGURES, BUILD, OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    check = subprocess.run(
        [sys.executable, str(PAPER / "verify_examples.py")],
        capture_output=True, text=True, check=True,
    )
    (PAPER / "evidence" / "illustrative_example_checks.json").write_text(check.stdout)
    plot_examples()
    subprocess.run(
        [sys.executable, str(PAPER / "evidence" / "check_evidence.py")],
        cwd=REPOSITORY, check=True,
    )
    subprocess.run(
        [sys.executable, str(PAPER / "generate_empirical_figures.py")],
        cwd=REPOSITORY, check=True,
    )
    if not args.figures_only:
        print(json.dumps({"pdf": str(compile_pdf())}, indent=2))


if __name__ == "__main__":
    main()
