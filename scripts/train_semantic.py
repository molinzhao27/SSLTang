#!/usr/bin/env python3
"""Fit continuous semantic regression from random or SSL encoder initialization."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config
from src.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    run_training(load_config(args.config, args.set), 'semantic', args.resume)


if __name__ == '__main__':
    main()
