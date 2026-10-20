#!/usr/bin/env python3
"""Discover exact public keys before fetching selected published derivatives."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.openneuro import discover,selection,download

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--subject',default='UTS01');p.add_argument('--mode',choices=['train','transfer'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--dry-run',action='store_true');p.add_argument('--manifest',help='Replay saved object keys/sizes/ETags instead of discovering current data');a=p.parse_args()
 dataset='ds003020' if a.mode=='train' else 'ds004510'
 if a.manifest:
  saved=json.loads(Path(a.manifest).read_text())
  if (saved['dataset'],saved['subject'],saved['mode'])!=(dataset,a.subject,a.mode):raise ValueError('Manifest dataset/subject/mode mismatch')
  selected=saved['objects']
  for item in selected:
   key=Path(item['key'])
   if key.is_absolute() or '..' in key.parts or not item['key'].startswith(dataset+'/'):raise ValueError('Invalid dataset key')
 else:selected=selection(discover(dataset),a.subject,a.mode)
 if not any(x['key'].endswith('.hf5') for x in selected):raise ValueError('No subject derivatives discovered')
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True);manifest={'dataset':dataset,'subject':a.subject,'mode':a.mode,'source':'https://s3.amazonaws.com/openneuro.org','objects':selected,'total_bytes':sum(x['size'] for x in selected)}
 (root/f'download_{dataset}_{a.subject}.json').write_text(json.dumps(manifest,indent=2));print(f"{len(selected)} objects, {manifest['total_bytes']/1e9:.3f} GB",flush=True)
 if not a.dry_run:
  with ThreadPoolExecutor(max_workers=a.workers) as pool:
   for path in pool.map(lambda x:download(x,root),selected):print(path,flush=True)
if __name__=='__main__':main()
