#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.story import evaluate_story

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate held-out ds003020 story runs")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--exclusion-tr", type=int, default=10)
    a = p.parse_args()
    evaluate_story(a.checkpoint, a.manifest, a.split, a.output_dir, a.device,
                   a.batch_size, a.num_workers, a.exclusion_tr)
