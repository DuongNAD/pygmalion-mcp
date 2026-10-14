#!/usr/bin/env python3
"""Aggregate driver model quality telemetry across CharForge projects.

Computes error_rate, iterations to pass, pass rate, total time, and error codes distribution.
Generates structured Markdown report (docs/reports/driver-<model>.md).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate CharForge driver model telemetry.")
    ap.add_argument("--projects-dir", default=None, help="Root directory containing project folders (default: projects/ or CHARFORGE_HOME)")
    ap.add_argument("--model", default="default", help="Driver model name identifier (e.g. gemini-flash, claude-sonnet)")
    ap.add_argument("--out", default=None, help="Output markdown path (default: docs/reports/driver-<model>.md)")
    return ap.parse_args()


def collect_project_metrics(pdir: Path) -> dict | None:
    pjson_path = pdir / "project.json"
    if not pjson_path.is_file():
        return None

    try:
        p = json.loads(pjson_path.read_text())
    except Exception:
        return None

    calls_path = pdir / "calls.jsonl"
    recs = []
    if calls_path.is_file():
        for line in calls_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass

    total_calls = len(recs)
    errors = [r for r in recs if not r.get("ok")]
    total_errors = len(errors)
    error_rate = total_errors / total_calls if total_calls > 0 else 0.0

    codes = {}
    for r in errors:
        c = r.get("code") or "UNKNOWN"
        codes[c] = codes.get(c, 0) + 1

    scores = p.get("scores", [])
    iterations = len(scores)
    best_total = (p.get("best") or {}).get("total")
    passed = p.get("stage") == "passed" or any(s.get("passed") for s in scores)
    total_sec = sum(float(r.get("sec", 0.0)) for r in recs)

    return {
        "id": p.get("id", pdir.name),
        "name": p.get("name", pdir.name),
        "stage": p.get("stage", "unknown"),
        "calls": total_calls,
        "errors": total_errors,
        "error_rate": error_rate,
        "error_codes": codes,
        "iterations": iterations,
        "best_total": best_total,
        "passed": passed,
        "total_sec": round(total_sec, 2),
    }


def generate_report(projects: list[dict], model_name: str, root_dir: Path) -> str:
    total_projs = len(projects)
    total_calls = sum(p["calls"] for p in projects)
    total_errors = sum(p["errors"] for p in projects)
    overall_error_rate = total_errors / total_calls if total_calls > 0 else 0.0

    passed_count = sum(1 for p in projects if p["passed"])
    pass_rate = passed_count / total_projs if total_projs > 0 else 0.0

    passed_iters = [p["iterations"] for p in projects if p["passed"]]
    avg_iters = sum(passed_iters) / len(passed_iters) if passed_iters else 0.0

    total_time = sum(p["total_sec"] for p in projects)

    # Aggregated error codes
    agg_codes = {}
    for p in projects:
        for c, count in p["error_codes"].items():
            agg_codes[c] = agg_codes.get(c, 0) + count

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# CharForge Driver Model Evaluation Report: {model_name}",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Evaluated At**: `{now_iso}`",
        f"- **Projects Directory**: `{root_dir}`",
        f"- **Projects Analyzed**: {total_projs}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Projects | {total_projs} |",
        f"| Total Tool Calls | {total_calls} |",
        f"| Total Errors | {total_errors} |",
        f"| Overall Error Rate | {overall_error_rate:.1%} |",
        f"| Pass Rate | {pass_rate:.1%} ({passed_count}/{total_projs}) |",
        f"| Avg Iterations to Pass | {avg_iters:.2f} |",
        f"| Total Time | {total_time:.2f}s |",
        "",
        "## Error Codes Distribution",
        "",
    ]

    if agg_codes:
        lines.extend([
            "| Error Code | Count | Share |",
            "|---|---|---|",
        ])
        for code, count in sorted(agg_codes.items(), key=lambda x: x[1], reverse=True):
            share = count / total_errors if total_errors > 0 else 0.0
            lines.append(f"| `{code}` | {count} | {share:.1%} |")
    else:
        lines.append("No errors recorded across all evaluated projects.")

    lines.extend([
        "",
        "## Project Breakdown",
        "",
        "| Project ID | Stage | Calls | Errors | Error Rate | Iterations | Best Total | Passed | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    if projects:
        for p in projects:
            best_str = f"{p['best_total']:.3f}" if p['best_total'] is not None else "N/A"
            passed_icon = "Yes" if p["passed"] else "No"
            lines.append(
                f"| `{p['id']}` | `{p['stage']}` | {p['calls']} | {p['errors']} | {p['error_rate']:.1%} | "
                f"{p['iterations']} | {best_str} | {passed_icon} | {p['total_sec']}s |"
            )
    else:
        lines.append("| *none* | - | 0 | 0 | 0.0% | 0 | N/A | No | 0.0s |")

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()

    # Determine projects directory
    if args.projects_dir:
        pdir = Path(args.projects_dir)
    elif os.environ.get("CHARFORGE_HOME"):
        pdir = Path(os.environ["CHARFORGE_HOME"])
    else:
        pdir = Path("projects")

    projects = []
    if pdir.is_dir():
        for child in sorted(pdir.iterdir()):
            if child.is_dir():
                metrics = collect_project_metrics(child)
                if metrics:
                    projects.append(metrics)

    report_md = generate_report(projects, args.model, pdir.resolve())

    # Determine output path
    out_path = Path(args.out) if args.out else Path("docs") / "reports" / f"driver-{args.model}.md"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md)
        print(f"Report written to: {out_path}")
    except Exception as e:
        print(f"Warning: could not write to {out_path}: {e}", file=sys.stderr)

    print("\n" + report_md)


if __name__ == "__main__":
    main()
