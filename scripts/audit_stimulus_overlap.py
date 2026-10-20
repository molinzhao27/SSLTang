#!/usr/bin/env python3
"""Check long verbatim reference overlap; never fits or reads neural arrays."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.semantic.features import read_word_intervals
from src.runtime import save_json, file_sha256

p=argparse.ArgumentParser();p.add_argument('--train-manifest',required=True);p.add_argument('--transfer-manifest',required=True);p.add_argument('--words',type=int,default=12);p.add_argument('--out',default='experiments/stimulus_overlap.json');a=p.parse_args()
index=defaultdict(set)
def words(path):
    return re.findall(r"[a-z0-9]+", " ".join(w for _,_,w in read_word_intervals(path)).lower())
train=json.loads(Path(a.train_manifest).read_text());transfer=json.loads(Path(a.transfer_manifest).read_text())
for run in train['runs']:
    tokens=words(run['stimulus_path'])
    for i in range(len(tokens)-a.words+1):index[tuple(tokens[i:i+a.words])].add(run['story'])
matches=[];seen=set()
for run in transfer['runs']:
    path=run.get('stimulus_path')
    if not path or path in seen:continue
    seen.add(path);tokens=words(path);shared=defaultdict(int)
    for i in range(len(tokens)-a.words+1):
        for story in index.get(tuple(tokens[i:i+a.words]),[]):shared[story]+=1
    matches.append({'transfer_story':run['story'],'text_sha256':file_sha256(path),'n_words':len(tokens),'training_stories_with_shared_ngrams':dict(shared)})
save_json({'n_gram_words':a.words,'matches':matches,'any_overlap':any(m['training_stories_with_shared_ngrams'] for m in matches),'interpretation':'No long verbatim match does not prove absence of paraphrase overlap or external GPT2 corpus contamination.'},a.out)
print(json.dumps({'any_overlap':any(m['training_stories_with_shared_ngrams'] for m in matches),'references':len(matches)}))
