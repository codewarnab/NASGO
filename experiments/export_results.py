#!/usr/bin/env python3
"""Export a NASGO SQLite experiment database to CSV, JSON, and Markdown.

Usage:
    python experiments/export_results.py --db experiments.db --out export_dir/

Produces:
    architectures.csv  - one row per evaluated architecture
    experiments.json   - experiment metadata (status, best fitness, counts)
    summary.md         - human-readable experiment summary

Only reads the database; safe to run on a live experiments.db copy.
"""
import argparse, csv, json, sqlite3
from pathlib import Path


def export(db_path, out_dir):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    experiments = [dict(r) for r in db.execute(
        "SELECT id, name, strategy, status, config_json, "
        "total_evaluations, best_fitness, started_at, completed_at FROM experiments")]
    (out / "experiments.json").write_text(json.dumps(experiments, indent=1, default=str))

    archs = [dict(r) for r in db.execute(
        "SELECT id, experiment_id, fitness, parameters, generation, "
        "mutation_type, evaluated_at, arch_json FROM architectures ORDER BY fitness DESC")]
    with (out / "architectures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "experiment_id", "fitness", "parameters",
                                          "generation", "mutation_type", "evaluated_at",
                                          "arch_json"])
        w.writeheader()
        w.writerows(archs)

    lines = ["# NASGO experiment export", ""]
    for e in experiments:
        lines.append(f"## {e['name']} ({e['id']})")
        lines.append(f"- strategy: {e['strategy']}  status: {e['status']}")
        lines.append(f"- evaluations: {e['total_evaluations']}")
        lines.append(f"- best fitness: {e['best_fitness']}")
        n = sum(1 for a in archs if a["experiment_id"] == e["id"])
        lines.append(f"- architecture rows exported: {n}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print(f"exported {len(experiments)} experiments, {len(archs)} architectures -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export(args.db, args.out)


if __name__ == "__main__":
    main()
