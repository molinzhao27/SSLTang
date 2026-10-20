#!/usr/bin/env python3
"""Export measured, coarse transfer comparisons without inventing precision."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',default='results/transfer');p.add_argument('--out',default='results/transfer_comparison');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rows=[]
for path in sorted(Path(a.root).glob('*/results.json')):
 m=json.loads(path.read_text());cfg=m['provenance']['training_config'];sem=cfg['semantic']
 for condition,result in m['conditions'].items():
  if result['status']!='scored':continue
  for space in ['raw','training_mean_centered']:
   score=result[space];macro=score['narrative_macro'];null=score['permutation']['metrics']
   rows.append({'subject':m['subject'],'fraction':sem['label_fraction'],'init':sem['init'],
    'seed':cfg['training']['seed'],'condition':condition,'space':space,
    'narratives':macro['n_narratives'],'chance_top1':result['top1_chance'],
    'top1':macro['top1'],'mrr':macro['mrr'],'mean_cosine':macro['mean_cosine'],
    'top1_exact_permutation_p':null['top1']['p_greater_equal'],
    'cosine_exact_permutation_p':null['mean_cosine']['p_greater_equal'],
    'constant_train_mean_raw_top1':result['constant_training_mean']['narrative_macro']['top1'],
    'source':str(path),'checkpoint_sha256':m['provenance']['checkpoint_sha256']})
if not rows:raise SystemExit('No scored transfer results; run evaluation first')
def csv_write(name,data):
 with (out/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");writer.writeheader();writer.writerows(data)
csv_write('per_seed.csv',rows)
pairs=[]
for base in rows:
 if base['init']!='random':continue
 same=[r for r in rows if r['init']=='ssl' and all(r[k]==base[k] for k in ['subject','fraction','seed','condition','space'])]
 if not same:continue
 ssl=same[0]
 pairs.append({k:base[k] for k in ['subject','fraction','seed','condition','space']}|
  {'delta_top1':ssl['top1']-base['top1'],'delta_mrr':ssl['mrr']-base['mrr'],'delta_cosine':ssl['mean_cosine']-base['mean_cosine'],
   'relative_top1_gain_percent':100*(ssl['top1']-base['top1'])/base['top1'] if base['top1']>0 else None})
csv_write('paired_effects.csv',pairs)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
lines=['# Held-out modality transfer','',
 'Frozen models; all training and model selection used ds003020. These are run-level reference matching scores, not recovered imagined words or time-aligned movie descriptions.', '',
 '| Condition | Training labels | Seeds per arm | Random centered top-1 | SSL centered top-1 | Chance |',
 '|---|---:|---:|---:|---:|---:|']
colors={'random':'#48677F','ssl':'#C47827'}
for subject in sorted({r['subject'] for r in rows}):
 fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
 for ax,condition,title in zip(axes,['imagined_speech','perceived_movie'],['Imagined speech · five cue references','Silent movies · three descriptions']):
  for arm in ['random','ssl']:
   selected=[r for r in rows if r['subject']==subject and r['condition']==condition and r['space']=='training_mean_centered' and r['init']==arm]
   fs=sorted({r['fraction'] for r in selected});groups=[[r['top1'] for r in selected if r['fraction']==f] for f in fs]
   means=np.array([np.mean(v) for v in groups]);low=np.array([np.min(v) for v in groups]);high=np.array([np.max(v) for v in groups])
   ax.errorbar(np.array(fs)*100,means*100,yerr=100*np.array([means-low,high-means]),fmt='o-' if arm=='ssl' else 's--',color=colors[arm],capsize=3,label=arm)
  chance=.2 if condition=='imagined_speech' else 1/3
  ax.axhline(chance*100,color='#666666',ls=':',lw=1,label='Chance')
  ax.set_xscale('log');ax.set_xticks([1,5,10,25,100],labels=['1','5','10','25','100'])
  ax.set(title=title,xlabel='Labeled training targets (%)',ylabel='Centered top-1 retrieval (%)',ylim=(0,100))
  ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=9)
  for fraction in [1,.25,.1,.05,.01]:
   arms={arm:[r for r in rows if r['subject']==subject and r['condition']==condition and r['space']=='training_mean_centered' and r['init']==arm and r['fraction']==fraction] for arm in ['random','ssl']}
   if not all(arms.values()):continue
   b,s=[100*np.mean([r['top1'] for r in arms[arm]]) for arm in ['random','ssl']]
   lines.append(f"| {condition} | {fraction*100:g}% | {len(arms['random'])}/{len(arms['ssl'])} | {b:.1f}% | {s:.1f}% | {chance*100:.1f}% |")
 fig.suptitle(f'{subject} · held-out ds004510 · means and observed seed ranges',fontsize=12)
 fig.savefig(out/f'{subject}_transfer.png',dpi=180);fig.savefig(out/f'{subject}_transfer.pdf');plt.close(fig)
lines += ['', 'Only five imagined narratives (two repeats each) and three scored films are available. Repeats move together in exact label permutations. Individual-model permutation p-values, raw-space retrieval, cosine, MRR and constant-training-mean controls are in per_seed.csv and the source JSON files. P-values are not averaged across seeds. Sintel is unscored because its reference transcript is absent. Error bars show seed ranges, not confidence intervals. This transfer is zero-shot with respect to neural training; external GPT2 corpus membership cannot be verified.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(out)
