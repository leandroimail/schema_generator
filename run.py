#!/usr/bin/env python3
"""Orchestrator for the schema_generator dictionary comparison pipeline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def cmd_validate(args) -> int:
    import glob
    cfg = load_config()
    data_llm_dir = PROJECT_ROOT / cfg.get("data_llm_results_dictionary_generation", {}).get("path", "")
    files = sorted(glob.glob(str(data_llm_dir / "**" / "*_parsed.json"), recursive=True))
    print(f"Found {len(files)} parsed JSON files under {data_llm_dir}")

    ok = err = 0
    err_samples = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and d.get("error"):
                err += 1
                if len(err_samples) < 10:
                    err_samples.append((p, str(d.get("error"))[:120]))
            else:
                ok += 1
        except Exception as e:
            err += 1
            if len(err_samples) < 10:
                err_samples.append((p, f"parse error: {e}"))

    print(f"OK:  {ok}")
    print(f"ERR: {err}")
    if err_samples:
        print("\nFirst errors:")
        for path, msg in err_samples:
            p = Path(path)
            try:
                display = p.relative_to(PROJECT_ROOT)
            except ValueError:
                display = p
            print(f"  {display}: {msg}")
    return 0


def cmd_retry_llm(args) -> int:
    py = find_python()
    cmd = [py, "-m", "src.dictionary_generation"]
    if args.dry_run:
        cmd.append("--list-errors")
    else:
        cmd.append("--retry-errors")
    print(">>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def cmd_compare(args) -> int:
    py = find_python()
    cmd = [py, "src/compare_results_dictionary.py"]
    print(">>>", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc == 0:
        summary_path = PROJECT_ROOT / "data" / "distance_calculation" / "all_similarities_results.json"
        if summary_path.exists():
            with open(summary_path) as f:
                d = json.load(f)
            print("\n=== Summary ===")
            print(f"Tables: {len(d.get('results', {}))}")
            avg = d.get("average_by_model", {})
            print(f"Models ({len(avg)}): {sorted(avg.keys())}")
            for m, s in sorted(avg.items()):
                print(f"  {m}: {s:.4f}")
    return rc


def cmd_all(args) -> int:
    rc = cmd_validate(args)
    if rc != 0:
        return rc
    if args.with_retry:
        rc = cmd_retry_llm(argparse.Namespace(dry_run=False))
        if rc != 0:
            return rc
    return cmd_compare(args)


def main() -> int:
    p = argparse.ArgumentParser(prog="run.py", description="Schema generator orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="Audit LLM result JSONs for errors").set_defaults(func=cmd_validate)

    retry = sub.add_parser("retry-llm", help="Rerun failed LLM dictionary generations")
    retry.add_argument("--dry-run", action="store_true", help="List targets without calling APIs")
    retry.set_defaults(func=cmd_retry_llm)

    sub.add_parser("compare", help="Run distance comparison against LLM dictionaries").set_defaults(func=cmd_compare)

    all_p = sub.add_parser("all", help="validate [+retry-llm] + compare")
    all_p.add_argument("--with-retry", action="store_true", help="Include retry-llm step")
    all_p.set_defaults(func=cmd_all)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
