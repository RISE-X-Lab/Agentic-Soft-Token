"""Render README benchmark graphics from the paper-reported results.

Optional authoring dependency: matplotlib. Run from any directory with
`python tools/render_readme_results.py`. Runtime inference does not use this tool.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MultipleLocator
except ImportError as exc:
    raise SystemExit("Install matplotlib to regenerate the README result graphics.") from exc

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("AppWorld", "OfficeBench", "8-Objective QA")
METHODS = ("ASTC", "Base")
# Accuracy/EM, F1, Decode KV, Attention Pairs, Generated Tokens (all in percent).
PAPER_RESULTS = {
    ("AppWorld", "Base"): (32.14, None, 100.00, 100.00, 100.00),
    ("AppWorld", "ASTC"): (33.33, None, 81.55, 80.70, 94.83),
    ("OfficeBench", "Base"): (48.42, None, 100.00, 100.00, 100.00),
    ("OfficeBench", "ASTC"): (47.37, None, 54.52, 49.32, 69.23),
    ("8-Objective QA", "Base"): (28.28, 37.34, 100.00, 100.00, 100.00),
    ("8-Objective QA", "ASTC"): (30.56, 38.46, 78.69, 98.51, 86.55),
}
METRICS = (
    ("quality", "Accuracy / EM (%) ↑", 0, "#CDEAC5", 162),
    ("decode", "Decode KV (%) ↓", 2, "#C6E4FF", 306),
    ("attention", "Attention pairs (%) ↓", 3, "#F5C3E2", 432),
    ("tokens", "Generated tokens (%) ↓", 4, "#FFD1A9", 558),
)
THEMES = {
    "light": {
        "background": "#FFFFFF",
        "foreground": "#294D73",
        "muted": "#60758C",
        "grid": "#DDE5EC",
        "base": "#CBD5DF",
        "highlight": "#F1F7F3",
    },
    "dark": {
        "background": "#0D1117",
        "foreground": "#E6EDF3",
        "muted": "#A8B8CA",
        "grid": "#303B46",
        "base": "#4B5B6D",
        "highlight": "#17282A",
    },
}
WIDTH, HEIGHT = 980, 686
COLUMN_X = (205, 455, 705)
PLOT_WIDTH, PLOT_HEIGHT = 184, 62
NS = "http://www.w3.org/2000/svg"


def read_results() -> dict[tuple[str, str], list[float | None]]:
    """Validate the six paper-reported rows used by both graphic themes."""
    rows = {key: list(values) for key, values in PAPER_RESULTS.items()}
    for key, values in rows.items():
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError(f"Invalid result value: {key}")
    if set(rows) != {(bench, method) for bench in BENCHMARKS for method in METHODS}:
        raise ValueError("Expected one Base and one ASTC row for each of the three benchmarks.")
    for bench in BENCHMARKS:
        if rows[(bench, "Base")][2:] != [100.0, 100.0, 100.0]:
            raise ValueError(f"Workload must use Base = 100%: {bench}")
    return rows


def slug(benchmark: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", benchmark.lower()).strip("-")


def draw_results(theme: str, rows: dict, preview_dir: Path | None) -> None:
    colors = THEMES[theme]
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "astc-readme-results",
            "axes.unicode_minus": False,
        }
    )
    figure = Figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=72, facecolor=colors["background"])

    def label(x, y, text, size=14, weight="normal", color=None, gid=None):
        return figure.text(
            x / WIDTH,
            1 - y / HEIGHT,
            text,
            fontsize=size,
            fontweight=weight,
            color=color or colors["foreground"],
            va="baseline",
            gid=gid,
        )

    label(28, 40, "Task quality and inference workload", 22, "bold")
    label(
        28,
        69,
        "Frozen Qwen3.5-9B · Paper-reported results · ASTC vs Base",
        14,
        color=colors["muted"],
    )
    for x, benchmark in zip(COLUMN_X, BENCHMARKS):
        label(x, 114, benchmark, 17, "bold")

    for metric, heading, value_index, astc_color, top in METRICS:
        label(28, top - 17, heading, 13, "bold")
        figure.add_artist(
            FancyBboxPatch(
                (16 / WIDTH, 1 - (top + 31) / HEIGHT),
                948 / WIDTH,
                31 / HEIGHT,
                boxstyle="round,pad=0,rounding_size=0.008",
                transform=figure.transFigure,
                facecolor=colors["highlight"],
                edgecolor="none",
                zorder=0,
            )
        )
        for method, center in (("ASTC", top + 15.5), ("Base", top + 46.5)):
            label(28, center + 5, method, 14, "bold" if method == "ASTC" else "normal")

        maximum = max(
            rows[(bench, method)][value_index] for bench in BENCHMARKS for method in METHODS
        )
        limit = math.ceil(max(maximum, 100 if metric != "quality" else 1) / 20) * 20
        step = 20 if metric == "quality" else (25 if limit == 100 else limit / 4)
        for x, benchmark in zip(COLUMN_X, BENCHMARKS):
            axis = figure.add_axes(
                [
                    x / WIDTH,
                    1 - (top + PLOT_HEIGHT) / HEIGHT,
                    PLOT_WIDTH / WIDTH,
                    PLOT_HEIGHT / HEIGHT,
                ],
                facecolor="none",
                zorder=1,
            )
            axis.set_xlim(0, limit)
            axis.set_ylim(-0.5, 1.5)
            axis.set_yticks([])
            axis.xaxis.set_major_locator(MultipleLocator(step))
            axis.tick_params(axis="x", labelsize=10, labelcolor=colors["muted"], length=0, pad=7)
            axis.set_axisbelow(True)
            axis.grid(axis="x", color=colors["grid"], linewidth=0.7)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for method, y in (("ASTC", 1), ("Base", 0)):
                value = rows[(benchmark, method)][value_index]
                group = f"{slug(benchmark)}-{metric}-{method.lower()}"
                bar = axis.barh(
                    y,
                    value,
                    height=0.62,
                    color=astc_color if method == "ASTC" else colors["base"],
                    zorder=3,
                )[0]
                bar.set_gid(group)
                axis.text(
                    value + limit * 0.045,
                    y,
                    f"{value:.2f}",
                    fontsize=12.5,
                    fontweight="bold" if method == "ASTC" else "normal",
                    va="center",
                    color=colors["foreground"],
                    clip_on=False,
                    gid="value-" + group,
                )
    label(
        COLUMN_X[2],
        259,
        f"F1: ASTC {rows[('8-Objective QA', 'ASTC')][1]:.2f} · "
        f"Base {rows[('8-Objective QA', 'Base')][1]:.2f}",
        11.5,
        color=colors["muted"],
        gid="qa-f1",
    )
    label(
        28,
        672,
        "Quality: higher is better. Workload: % of Base (100%); lower is better.",
        12,
        color=colors["muted"],
    )

    target = ROOT / "assets" / f"benchmark-results-{theme}.svg"
    figure.savefig(target, format="svg", metadata={"Date": None, "Creator": "Agentic Soft Token"})
    tree = ET.parse(target)
    svg = tree.getroot()
    svg.set("role", "img")
    svg.set("aria-labelledby", "results-title results-description")
    title = ET.Element(f"{{{NS}}}title", id="results-title")
    title.text = "Base and ASTC across three agent benchmarks"
    description = ET.Element(f"{{{NS}}}desc", id="results-description")
    parts = []
    for (benchmark, method), values in rows.items():
        parts.append(
            f"{benchmark}, {method}: Accuracy/EM {values[0]:.2f}%; "
            + (f"F1 {values[1]:.2f}%; " if values[1] is not None else "")
            + f"Decode KV {values[2]:.2f}%; Attention Pairs {values[3]:.2f}%; "
            + f"Generated Tokens {values[4]:.2f}%."
        )
    description.text = " ".join(parts)
    svg.insert(0, title)
    svg.insert(1, description)
    groups = {element.get("id"): element for element in svg.iter() if element.get("id")}
    for benchmark in BENCHMARKS:
        for metric, _, index, _, _ in METRICS:
            for method in METHODS:
                group = f"{slug(benchmark)}-{metric}-{method.lower()}"
                groups[group].set("data-value", f"{rows[(benchmark, method)][index]:.2f}")
    ET.register_namespace("", NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree.write(target, encoding="unicode", xml_declaration=False)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        FigureCanvasAgg(figure)
        figure.savefig(preview_dir / f"benchmark-results-{theme}.png", dpi=144)
    print(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    rows = read_results()
    for theme in THEMES:
        draw_results(theme, rows, args.preview_dir)


if __name__ == "__main__":
    main()
