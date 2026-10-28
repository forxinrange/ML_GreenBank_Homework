from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bank_ctf", description="Green Bank Pipeline CTF participant CLI")
    sub = p.add_subparsers(dest="command", required=True)
    
    train = sub.add_parser("train")
    train.add_argument("--task", choices=["inflow", "risk"], required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    
    pred = sub.add_parser("predict")
    pred.add_argument("--task", choices=["inflow", "risk"], required=True)
    pred.add_argument("--model-dir", required=True)
    pred.add_argument("--data-dir", required=True)
    pred.add_argument("--output", required=True)
    
    exp = sub.add_parser("experiment")
    exp.add_argument("--config", required=True)
    exp.add_argument("--data-dir", required=True)
    exp.add_argument("--runs-dir", required=True)
    exp.add_argument("--run-id", required=True)
    return p


def read_config(path: str | Path) -> dict[str, Any]:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {path}: {exc}") from exc
    if cfg.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    return cfg


def run_experiment(config_path: str, data_dir: str, runs_dir: str, run_id: str):
    cfg = read_config(config_path)
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    log_file = runs_path / "runs.jsonl"

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if json.loads(line).get("run_id") == run_id:
                    raise ValueError(f"Run ID '{run_id}' already exists in {log_file}")

    record = {
        "run_id": run_id,
        "config": cfg,
        "status": "success",
        "data_dir": str(data_dir)
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Experiment {run_id} logged.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            if args.task == "inflow":
                pipeline.train_inflow(Path(args.data_dir), Path(args.output_dir), Path(args.config))
            elif args.task == "risk":
                pipeline.train_risk(Path(args.data_dir), Path(args.output_dir), Path(args.config))
            return 0
            
        elif args.command == "predict":
            if args.task == "inflow":
                pipeline.predict_inflow(Path(args.data_dir), Path(args.model_dir), Path(args.output))
            elif args.task == "risk":
                pipeline.predict_risk(Path(args.data_dir), Path(args.model_dir), Path(args.output))
            return 0
            
        elif args.command == "experiment":
            run_experiment(args.config, args.data_dir, args.runs_dir, args.run_id)
            return 0
            
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())