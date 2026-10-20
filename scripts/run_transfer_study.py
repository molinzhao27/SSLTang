#!/usr/bin/env python3
"""Evaluate every completed semantic model once; never train or tune on transfer."""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import file_sha256
p=argparse.ArgumentParser();p.add_argument('--runs',default='results/runs');p.add_argument('--manifest',default='data/transfer/UTS01/manifest.json');p.add_argument('--out',default='results/transfer');p.add_argument('--config',default='configs/transfer_uts01.yaml');a=p.parse_args()
checkpoints=sorted(Path(a.runs).glob('*_fraction*_seed*/best.pt'))
if not checkpoints:raise SystemExit('No semantic checkpoints found')
for checkpoint in checkpoints:
 if not (checkpoint.parent/'completed.json').exists():raise SystemExit('Finish supervised training first: '+str(checkpoint.parent))
for checkpoint in checkpoints:
 output=Path(a.out)/checkpoint.parent.name
 if (output/'results.json').exists():
  old=json.loads((output/'results.json').read_text())
  if old['provenance']['checkpoint_sha256']!=file_sha256(checkpoint):raise ValueError('Checkpoint differs from existing transfer result; use a new output directory')
  continue
 subprocess.run([sys.executable,'scripts/evaluate_transfer.py','--config',a.config,'--set',f'evaluation.manifest={a.manifest}','--set',f'evaluation.checkpoint={checkpoint}','--set',f'evaluation.output_dir={output}','--set','evaluation.feature_batch_size=256'],check=True)
