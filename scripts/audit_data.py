#!/usr/bin/env python3
"""Report real-data split, normalization, source hashes and nested-label counts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.windows import NeuralWindows,SemanticWindows
from src.runtime import save_json,file_sha256

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--semantic-manifest');p.add_argument('--out',default='experiments/data_validation.json');a=p.parse_args()
m=json.loads(Path(a.manifest).read_text()); report={'manifest_sha256':file_sha256(a.manifest),'normalization_sha256':file_sha256(m['normalization_path']),'splits':{}}
train_sum=train_sq=None;train_n=0
for split in ['train','val','test']:
 d=NeuralWindows(a.manifest,split);runs=d.runs
 for r in runs:
  x=np.load(r['neural_path'],mmap_mode='r')
  if not np.isfinite(x).all():raise ValueError('Nonfinite neural array: '+r['run_id'])
  if split=='train':
   sums=x.sum(0,dtype=np.float64);sq=np.square(x,dtype=np.float64).sum(0)
   train_sum=sums if train_sum is None else train_sum+sums
   train_sq=sq if train_sq is None else train_sq+sq;train_n+=len(x)
 report['splits'][split]={'runs':len(runs),'sessions':sorted({r['session'] for r in runs}),'trs':sum(r['n_tr'] for r in runs),'window_examples':len(d)}
report['normalized_training_max_abs_mean']=float(np.max(abs(train_sum/train_n)))
report['normalized_training_std_range']=[float(v) for v in [np.sqrt(train_sq/train_n).min(),np.sqrt(train_sq/train_n).max()]]
if a.semantic_manifest:
 subsets={};previous=set(); counts=[]
 val=SemanticWindows(a.semantic_manifest,'val');total=SemanticWindows(a.semantic_manifest,'train')
 for fraction in [.01,.05,.1,.25,1.]:
  d=SemanticWindows(a.semantic_manifest,'train',label_fraction=fraction);current=set(d.indices)
  if not previous<=current:raise ValueError('Label subsets not nested')
  previous=current;meta=dict(d.label_metadata);meta.pop('label_ids');meta['validation_labels']=len(val);meta['total_labels_including_validation']=len(d)+len(val);subsets[str(fraction)]=meta
 report['label_budgets']=subsets
save_json(report,a.out);print(json.dumps(report,indent=2))
