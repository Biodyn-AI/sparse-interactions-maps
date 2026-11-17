#!/usr/bin/env python3
"""Render audited saved plot values without changing original scientific files."""

import importlib.util
import json
from pathlib import Path

from check_repository import check


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    check(root)
    paper = root / "papers/sparse-interaction-maps"
    output = root / "output/figures"
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads((paper / "evidence/figure_source_data.json").read_text())
    empirical = load_module("saved_empirical_plotting", paper / "generate_empirical_figures.py")
    empirical.FIGURES = output
    empirical.style()
    empirical.historical_figure(data)
    empirical.fresh_figure(data)
    empirical.write_e32_2_table(data)
    examples = load_module("saved_example_plotting", paper / "build_paper.py")
    examples.FIGURES = output
    examples.plot_examples()
    print(f"Rendered three figures and a table from saved audited values to {output}; no model calls.")


if __name__ == "__main__":
    main()
