#!/usr/bin/env python3
"""Evaluate ds004510 with a frozen checkpoint and coarse run-level references."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config
from src.evaluation.transfer import run_transfer_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    result = run_transfer_evaluation(load_config(args.config, args.set))
    print(result, flush=True)


if __name__ == '__main__':
    main()
