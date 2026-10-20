#!/usr/bin/env python3
"""Spatially pool published H5 data; fit neural normalization on train only."""
import argparse,json,sys,re,hashlib
from pathlib import Path
import numpy as np,h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def groups(mask_path,block):
 import nibabel as nib
 # pycortex mask convention is [z,y,x], then C-order boolean indexing.
 mask=nib.load(str(mask_path)).get_fdata().transpose(2,1,0)>0
 coords=np.argwhere(mask);bins=coords//block
 _,membership=np.unique(bins,axis=0,return_inverse=True)
 return membership,coords

def pooled(path,membership):
 with h5py.File(path,'r') as f:
  data=f['data']
  if data.ndim!=2 or data.shape[1]!=len(membership):raise ValueError(f'Voxel dimension mismatch {path}: {data.shape}')
  n=int(membership.max())+1;counts=np.bincount(membership,minlength=n);out=np.empty((len(data),n),np.float32)
  for t in range(len(data)):
   row=data[t]; row=np.nan_to_num(row,nan=0,posinf=0,neginf=0)
   out[t]=np.bincount(membership,weights=row,minlength=n)/counts
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--subject',default='UTS01');p.add_argument('--mode',choices=['train','transfer'],required=True);p.add_argument('--normalization',help='Required for transfer; train-produced npz');p.add_argument('--block-size',type=int,default=4);p.add_argument('--session-map',default=str(Path(__file__).resolve().parents[1]/'configs/huth_session_map.json'));p.add_argument('--seed',type=int,default=42);p.add_argument('--val-fraction',type=float,default=.15);p.add_argument('--test-fraction',type=float,default=.15);a=p.parse_args()
 root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 if a.block_size<1:raise ValueError('block-size must be positive')
 if a.mode=='train' and not (0<a.val_fraction<1 and 0<a.test_fraction<1 and a.val_fraction+a.test_fraction<1):raise ValueError('Invalid split fractions')
 dataset='ds003020' if a.mode=='train' else 'ds004510';source=json.loads((root/f'download_{dataset}_{a.subject}.json').read_text());keys=[x['key'] for x in source['objects']]
 paths=[root/k for k in keys if '/preprocessed_data/'+a.subject+'/' in k and k.endswith('.hf5')];paths=sorted(paths)
 if not paths:raise ValueError('No neural runs')
 sessions={}
 for key in keys:
  if key.endswith('_bold.json'):
   meta=json.loads((root/key).read_text())
   if 'RepetitionTime' not in meta or float(meta['RepetitionTime'])!=2:raise ValueError('Missing or unexpected source TR: '+key)
  m=re.search(r'/ses-([^/]+)/.*_task-([^_]+).*_bold.json$',key)
  if m:
   sessions.setdefault(m[2].lower(),set()).add(m[1])
 if a.mode=='train':
  # Extended sessions 7-20 are absent from the original BIDS sidecars.
  # This author-supplied map was fetched from semantic-decoding README's data_train archive.
  for session, stories in json.loads(Path(a.session_map).read_text()).items():
   for story in stories:
    if story in sessions and sessions[story]!={session}:raise ValueError('Conflicting author/BIDS session for '+story)
    sessions.setdefault(story,set()).add(session)
  missing=[p.stem for p in paths if p.stem not in sessions]
  if missing:raise ValueError('Missing acquisition session metadata: '+str(missing))
  masks=[root/k for k in keys if k.endswith('mask_thick.nii.gz')]
  if len(masks)!=1:raise ValueError('Expected one discovered subject thick mask; inspect subject_xfms.json: '+str(masks))
  membership,coords=groups(masks[0],a.block_size)
  # Whole acquisition sessions stay together. Unknown session is a distinct story group.
  run_groups={p.stem:','.join(sorted(sessions.get(p.stem.lower(),{p.stem}))) for p in paths}
  # Connected components prevent any shared session from crossing a split.
  sets={p.stem:set(sessions.get(p.stem.lower(),{'story:'+p.stem})) for p in paths}
  changed=True
  while changed:
   changed=False
   for name in sets:
    for other in sets:
     if sets[name]&sets[other] and sets[name]!=sets[other]:
      union=sets[name]|sets[other];sets[name]=union;sets[other]=union;changed=True
  run_groups={name:','.join(sorted(value)) for name,value in sets.items()}
  unique=sorted(set(run_groups.values()));rng=np.random.default_rng(a.seed);rng.shuffle(unique)
  nv=max(1,round(len(unique)*a.val_fraction));nt=max(1,round(len(unique)*a.test_fraction))
  assignment={s:('val' if i<nv else 'test' if i<nv+nt else 'train') for i,s in enumerate(unique)}
 else:
  if not a.normalization:raise ValueError('--normalization is required for transfer')
  state=np.load(a.normalization);membership=state['membership'];coords=state['coords']
  if str(state['subject'])!=a.subject:raise ValueError('Subject mismatch')
 runs=[];total=None;squares=None;count=0
 for path in paths:
  story=path.stem.split('_repeat-')[0];condition='perceived_speech' if a.mode=='train' else path.parent.name
  split=assignment[run_groups[path.stem]] if a.mode=='train' else 'transfer'
  x=pooled(path,membership)
  if a.mode=='train':
   counts=json.loads(next(root/k for k in keys if k.endswith('respdict.json')).read_text())
   if len(x)!=counts[story]-20:raise ValueError(f'Timing length mismatch {story}: {len(x)} vs {counts[story]-20}')
  run_id=f'{dataset}_{a.subject}_{condition}_{path.stem}';dest=out/(run_id+'.npy');np.save(dest,x)
  if split=='train':
   sums=x.sum(axis=0,dtype=np.float64);sq=np.square(x,dtype=np.float64).sum(axis=0)
   total=sums if total is None else total+sums;squares=sq if squares is None else squares+sq;count+=len(x)
  grids=[root/k for k in keys if '/TextGrids/' in k and Path(k).stem==story]
  r={'run_id':run_id,'dataset':dataset,'subject':a.subject,'condition':condition,'session':','.join(sorted(sessions.get(story.lower(),[]))) or None,'split':split,'neural_path':str(dest),'n_tr':len(x),'n_features':x.shape[1],'tr_seconds':2.0,'tr_start_seconds':11.0 if a.mode=='train' else None,'stimulus_timing_verified':a.mode=='train','story':story,'source_key':str(path.relative_to(root)),'repeat':int(path.stem.split('_repeat-')[1]) if '_repeat-' in path.stem else None}
  if grids:r['stimulus_path']=str(grids[0])
  else:r['stimulus_missing_reason']='No published TextGrid in discovered dataset tree'
  runs.append(r);print(run_id,x.shape,split,flush=True)
 if a.mode=='train':
  if count==0:raise ValueError('No train rows')
  mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,0));std[std<1e-8]=1
  norm=out/'normalization.npz';np.savez(norm,mean=mean.astype(np.float32),std=std.astype(np.float32),membership=membership,coords=coords,subject=a.subject,fit_run_ids=np.array([r['run_id'] for r in runs if r['split']=='train']),n_fit_tr=count,block_size=a.block_size)
 else:mean=state['mean'];std=state['std'];norm=Path(a.normalization).resolve()
 for r in runs:
  f=Path(r['neural_path']);x=np.load(f);np.save(f,((x-mean)/std).astype(np.float32))
 manifest={'schema_version':1,'subject':a.subject,'mode':a.mode,'normalization_path':str(norm),'normalization_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),'normalized':True,'split_seed':a.seed,'source_manifest_sha256':hashlib.sha256((root/f'download_{dataset}_{a.subject}.json').read_bytes()).hexdigest(),'session_map_sha256':hashlib.sha256(Path(a.session_map).read_bytes()).hexdigest() if a.mode=='train' else None,'feature_order':'pycortex thick-mask C-order [z,y,x], spatial bins lexicographic','coordinate_identity_evidence':'Huth published train/test code applies same train-derived voxel indices; no per-column coordinates in H5','runs':runs}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
