"""Command-line interface for Evidence-k.

Subcommands:

* ``evidence-k init``            — scaffold an example config + datasets
* ``evidence-k run``             — run the k-sweep benchmark
* ``evidence-k summarize``       — print a saved run's summary
* ``evidence-k export-profile``  — (re)export ``k_profile.json`` from a run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config
from .outputs.profile import build_profile
from .outputs.reports import build_summary, write_run
from .runners.sweep import run_sweep
from .scaffold import build_example_datasets, example_config_yaml
from .utils.jsonl import write_jsonl


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


# --- init --------------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir)
    cfg_path = target / "configs" / "example.yaml"
    created: list[str] = []
    skipped: list[str] = []

    def _write(path: Path, content: str) -> None:
        if path.exists() and not args.force:
            skipped.append(str(path))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(str(path))

    _write(cfg_path, example_config_yaml())
    for rel, rows in build_example_datasets().items():
        path = target / rel
        if path.exists() and not args.force:
            skipped.append(str(path))
            continue
        write_jsonl(path, rows)
        created.append(str(path))

    (target / "runs").mkdir(parents=True, exist_ok=True)

    for c in created:
        print(f"created  {c}")
    for s in skipped:
        print(f"skipped  {s} (exists; use --force to overwrite)")
    print("\nNext: evidence-k run --config configs/example.yaml")
    return 0


# --- run ---------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.config).resolve().parent.parent
    result = run_sweep(cfg, base_dir=base_dir)
    run_dir = write_run(result, runs_root=Path(args.runs_dir))

    print(f"\nRun complete: {result.run_id}")
    print(f"Artifacts:    {run_dir}")
    _print_summary(build_summary(result))
    print(f"\nk-profile:    {run_dir / 'k_profile.json'}")
    return 0


# --- summarize ---------------------------------------------------------------------------


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary.json found in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def cmd_summarize(args: argparse.Namespace) -> int:
    summary = _load_summary(Path(args.run_dir))
    _print_summary(summary)
    return 0


# --- export-profile ----------------------------------------------------------------------


def cmd_export_profile(args: argparse.Namespace) -> int:
    summary = _load_summary(Path(args.run_dir))
    profile = build_profile(summary)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote k-profile: {out}")
    print(f"  model={profile['model']} provider={profile['provider']}")
    print(f"  global default_k={profile['global_recommendation'].get('default_k')}")
    return 0


# --- shared print ------------------------------------------------------------------------


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nModel: {summary['model']} (provider: {summary['provider']})")
    print(f"k values: {summary['tested_k_values']}")
    tested = summary["tested_k_values"]
    for task_name, tsum in summary.get("tasks", {}).items():
        per_k = tsum.get("per_k", {})
        print(f"\n  {task_name}  → recommended k = {tsum['recommended_k']} "
              f"(reliability {tsum['best_score']:.3f})")
        print(f"    {'k':>5} | {'reliab':>7} | {'correct':>7} | {'ground':>7} | "
              f"{'halluc':>7} | {'var':>6}")
        for k in tested:
            kl = "full" if k == "full" else str(k)
            if kl not in per_k:
                continue
            m = per_k[kl]
            print(f"    {kl:>5} | {m['reliability']:>7.3f} | {m['correctness']:>7.3f} | "
                  f"{m['grounding']:>7.3f} | {m['hallucination_rate']:>7.3f} | "
                  f"{m['answer_variance']:>6.3f}")
    gr = summary.get("global_recommendation", {})
    print(f"\n  Global default_k = {gr.get('default_k')} — {gr.get('notes', '')}")
    dual = summary.get("dual")
    if dual:
        from .outputs.dual_report import format_dual_report

        print(format_dual_report(dual))


# --- parser ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evidence-k",
        description="Measure how much evidence an LLM can use before context turns into noise.",
    )
    parser.add_argument("--version", action="version", version=f"evidence-k {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="scaffold an example config + datasets")
    p_init.add_argument("--dir", default=".", help="target directory (default: .)")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="run the k-sweep benchmark")
    p_run.add_argument("--config", required=True, help="path to a YAML config")
    p_run.add_argument(
        "--base-dir",
        default=None,
        help="base directory for resolving relative dataset paths (default: config's repo root)",
    )
    p_run.add_argument("--runs-dir", default="runs", help="where to write runs (default: runs/)")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summarize", help="print a saved run's summary")
    p_sum.add_argument("--run-dir", required=True, help="path to runs/<run_id>")
    p_sum.set_defaults(func=cmd_summarize)

    p_exp = sub.add_parser("export-profile", help="(re)export k_profile.json from a run")
    p_exp.add_argument("--run-dir", required=True, help="path to runs/<run_id>")
    p_exp.add_argument("--out", required=True, help="output path for the k-profile JSON")
    p_exp.set_defaults(func=cmd_export_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - surface a clear message, never a silent crash
        _eprint(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
