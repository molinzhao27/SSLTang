#!/usr/bin/env python3
"""Sequential GPU jobs with atomic per-run checkpoints and automatic resume.

Stage milestone runs 100% labels; labels runs the remaining four fractions.
Configs are materialized before fitting. This script never opens ds004510.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, save_json, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["ssl", "milestone", "labels"], required=True)
    p.add_argument("--ssl-config", default="configs/ssl_uts01.yaml")
    p.add_argument("--semantic-config", default="configs/semantic_uts01.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 47])
    p.add_argument("--root", default="results/runs")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    root = Path(a.root)
    for seed in a.seeds:
        if a.stage == "ssl":
            jobs = [("ssl", None)]
        else:
            fractions = [1.0] if a.stage == "milestone" else [0.25, 0.10, 0.05, 0.01]
            jobs = [(arm, fraction) for fraction in fractions for arm in ["random", "ssl"]]
        for arm, fraction in jobs:
            cfg = load_config(a.ssl_config if fraction is None else a.semantic_config, a.set)
            cfg["training"]["seed"] = seed
            subject = json.loads(Path(cfg["data"]["manifest"]).read_text())["runs"][0]["subject"]
            name = f"ssl_{subject}_seed{seed}" if fraction is None else f"{arm}_{subject}_fraction{fraction:g}_seed{seed}"
            out = root/name
            cfg["output_dir"] = str(out)
            if fraction is not None:
                cfg["semantic"].update(init=arm, label_fraction=fraction,
                    pretrained_encoder=str(root/f"ssl_{subject}_seed{seed}"/"encoder_best.pt"))
            out.mkdir(parents=True, exist_ok=True)
            config_path = out/"config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != cfg:
                raise ValueError(f"Existing experiment has different config: {out}; choose a new root")
            save_json(cfg, config_path)
            metadata_path = out/"run_metadata.json"
            if metadata_path.exists() and json.loads(metadata_path.read_text())["manifest_sha256"] != file_sha256(cfg["data"]["manifest"]):
                raise ValueError(f"Dataset/split manifest changed for {out}; use a new results root")
            done = out/"completed.json"
            if not done.exists():
                command = [sys.executable, "scripts/pretrain_ssl.py" if fraction is None else "scripts/train_semantic.py",
                           "--config", str(config_path)]
                if (out/"latest.pt").exists():
                    command += ["--resume", "latest"]
                subprocess.run(command, check=True)
                # SIGTERM can return cleanly after checkpoint; only mark full budgets complete.
                import torch
                checkpoint = torch.load(out/"latest.pt", map_location="cpu", weights_only=False)
                if checkpoint["epoch"] < cfg["training"]["epochs"] and checkpoint["global_step"] < cfg["training"].get("max_steps", float("inf")):
                    raise SystemExit("Training interrupted; rerun the same command to resume")
                save_json({"checkpoint": str(out/"best.pt"), "global_step": checkpoint["global_step"]}, done)
            if fraction is not None and not (out/"evaluation_test"/"metrics.json").exists():
                subprocess.run([sys.executable, "scripts/evaluate.py", "--checkpoint", str(out/"best.pt")], check=True)


if __name__ == "__main__":
    main()
