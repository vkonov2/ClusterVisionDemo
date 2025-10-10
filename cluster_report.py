"""Generate textual reports to inspect clustering results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

DEFAULT_MANIFEST = Path("data/embeddings/cluster_manifest.json")
DEFAULT_SUMMARY_CSV = Path("data/embeddings/cluster_summary.csv")
DEFAULT_REPORT_HTML = Path("data/embeddings/cluster_report.html")


def load_manifest(path: Path) -> Tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster manifest not found: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Manifest does not contain an 'entries' list")

    frame = pd.DataFrame(entries)
    if "label" not in frame.columns:
        raise ValueError("Manifest entries must contain a 'label' column")

    return frame, manifest


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("label").size().rename("count").reset_index()
    total = counts["count"].sum()
    counts["share"] = counts["count"] / total if total else 0.0
    counts["percentage"] = counts["share"] * 100
    return counts.sort_values("label").reset_index(drop=True)


def render_section(title: str, body: str) -> str:
    return f"<section><h2>{title}</h2>{body}</section>"


def dataframe_to_html(df: pd.DataFrame, classes: Iterable[str] | None = None) -> str:
    return df.to_html(index=False, escape=False, border=0, classes=list(classes or ()))


def build_report(summary: pd.DataFrame, df: pd.DataFrame, manifest: dict) -> str:
    parts = ["<html><head><meta charset='utf-8'>",
             "<style>body{font-family:Arial,sans-serif;margin:2rem;}"
             "table{border-collapse:collapse;margin-bottom:2rem;}"
             "th,td{border:1px solid #ddd;padding:0.4rem 0.6rem;}"
             "th{background:#f0f0f0;}"
             "h1,h2{margin-top:1.2rem;}"
             "</style></head><body>"]
    parts.append("<h1>Cluster report</h1>")
    parts.append("<p><strong>Algorithm:</strong> {}</p>".format(manifest.get("algorithm", "unknown")))
    if "inertia" in manifest:
        parts.append("<p><strong>Inertia:</strong> {:.4f}</p>".format(manifest["inertia"]))
    summary_display = summary.copy()
    summary_display["share"] = summary_display["share"].map(lambda x: f"{x:.4f}")
    summary_display["percentage"] = summary_display["percentage"].map(lambda x: f"{x:.2f}%")
    parts.append(render_section("Cluster summary", dataframe_to_html(summary_display)))

    for label, group in sorted(df.groupby("label"), key=lambda item: item[0]):
        if "index" in group.columns:
            group_sorted = group.sort_values(by="index")
        else:
            group_sorted = group.sort_values(by=[col for col in group.columns if col != "label"])
        parts.append(render_section(f"Cluster {label}", dataframe_to_html(group_sorted)))

    parts.append("</body></html>")
    return "".join(parts)


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_report(manifest_path: Path, summary_csv: Path, report_html: Path) -> None:
    df, manifest = load_manifest(manifest_path)
    summary = compute_summary(df)

    ensure_parent(summary_csv)
    summary.to_csv(summary_csv, index=False)

    ensure_parent(report_html)
    report_html.write_text(build_report(summary, df, manifest), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reports for clustering results")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to cluster_manifest.json")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV, help="Where to write cluster summary CSV")
    parser.add_argument("--report-html", type=Path, default=DEFAULT_REPORT_HTML, help="Where to write cluster report HTML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(args.manifest, args.summary_csv, args.report_html)


if __name__ == "__main__":
    main()
