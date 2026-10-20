"""Run-bounded neural windows and fixed, nested contiguous label subsets.

The neural dataset never opens a transcript or target file. No example crosses
a run boundary. Label fractions count eligible supervised target TRs, not SSL
examples; unlabeled context is allowed equally in both initialization arms.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_path(path, manifest_path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return Path(manifest_path).resolve().parent / path


class NeuralWindows(Dataset):
    def __init__(self, manifest_path, split, window=32, stride=4):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.split, self.window, self.stride = split, int(window), int(stride)
        if self.window < 1 or self.stride < 1:
            raise ValueError("Window and stride must be positive")
        all_runs = self.manifest["runs"]
        if split in {"train", "val", "test"} and any(
                r["dataset"] != "ds003020" for r in all_runs):
            raise ValueError("Training/development manifests must contain only ds003020")
        self.runs = [r for r in all_runs if r["split"] == split]
        if not self.runs:
            raise ValueError(f"No {split} runs in {manifest_path}")
        self._x, self._y, self._valid = {}, {}, {}
        self.n_features = int(self.runs[0]["n_features"])
        if any(int(r["n_features"]) != self.n_features for r in all_runs):
            raise ValueError("All runs must share the same spatial representation")
        if len({r["subject"] for r in all_runs}) != 1:
            raise ValueError("Fit separate subject-specific encoders; cross-subject alignment is not implemented")
        # Prevent duplicate identities crossing partitions, including repetitions.
        partitions = {}
        for r in all_runs:
            stimulus = r.get("story") or r.get("stimulus_id") or r["run_id"]
            identities = [("stimulus", r["subject"], stimulus),
                          ("run", r["subject"], r["run_id"]),
                          ("array", str(resolve_path(r["neural_path"], self.manifest_path).resolve()))]
            sessions = r.get("session") or []
            if isinstance(sessions, str):
                sessions = [s.strip() for s in sessions.split(",") if s.strip()]
            identities.extend(("session", r["subject"], session) for session in sessions)
            for identity in identities:
                if identity in partitions and partitions[identity] != r["split"]:
                    raise ValueError(f"Repeated stimulus/run/session crosses splits: {identity}")
                partitions[identity] = r["split"]
        normalization = self.manifest.get("normalization_path")
        if split in {"train", "val", "test"} and normalization:
            with np.load(resolve_path(normalization, self.manifest_path)) as state:
                fit_ids = set(map(str, state["fit_run_ids"]))
                held_out = {r["run_id"] for r in all_runs if r["split"] != "train"}
                if not fit_ids or fit_ids & held_out:
                    raise ValueError("Neural normalization must be fit on training runs only")
        self.indices = [(i, end) for i, r in enumerate(self.runs)
                        for end in range(self.window - 1, int(r["n_tr"]), self.stride)]
        if not self.indices:
            raise ValueError("No full neural windows; shorten window or inspect run durations")

    def _array(self, run_index, kind="neural_path"):
        cache = self._x if kind == "neural_path" else self._y if kind == "target_path" else self._valid
        if run_index not in cache:
            run = self.runs[run_index]
            values = np.load(resolve_path(run[kind], self.manifest_path), mmap_mode="r")
            if kind == "neural_path":
                expected = (int(run["n_tr"]), self.n_features)
                if values.shape != expected:
                    raise ValueError(f"Neural shape differs from manifest: {run['run_id']}")
            elif kind == "target_path":
                if values.ndim != 2 or values.shape[0] != int(run["n_tr"]) or values.shape[1] < 1:
                    raise ValueError(f"Targets must align with run TRs: {run['run_id']}")
            elif values.shape != (int(run["n_tr"]),) or values.dtype != np.bool_:
                raise ValueError(f"Validity mask must be one boolean per TR: {run['run_id']}")
            cache[run_index] = values
        return cache[run_index]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ri, end = self.indices[index]
        x = np.array(self._array(ri)[end-self.window+1:end+1], dtype=np.float32, copy=True)
        return {"x": torch.from_numpy(x), "run_index": ri, "end_tr": end,
                "run_id": self.runs[ri]["run_id"]}

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_x={}, _y={}, _valid={})
        return state


class SemanticWindows(NeuralWindows):
    def __init__(self, manifest_path, split, window=32, stride=4,
                 label_fraction=1.0, subset_seed=2718, lag_tr=3, block_tr=64):
        super().__init__(manifest_path, split, window, stride)
        self.lag_tr = int(lag_tr)
        if self.lag_tr < 0:
            raise ValueError("Neural responses must follow semantic targets")
        if not 0 < label_fraction <= 1:
            raise ValueError("label_fraction must be in (0,1]")
        block_tr = int(block_tr)
        if block_tr < 1:
            raise ValueError("block_tr must be positive")
        for r in self.runs:
            if "target_path" not in r or "target_valid_path" not in r:
                raise ValueError(f"Run lacks verified semantic targets: {r['run_id']}")
        self.target_dim = int(self._array(0, "target_path").shape[-1])
        if any(self._array(ri, "target_path").shape[1] != self.target_dim for ri in range(len(self.runs))):
            raise ValueError("All runs must share the same semantic target dimension")
        candidates = [(ri, end) for ri, end in self.indices if end >= self.lag_tr
                      and end-self.lag_tr < len(self._array(ri, "target_path"))
                      and bool(self._array(ri, "target_valid_path")[end-self.lag_tr])]
        # Randomize entire temporal blocks, then retain their internal order.
        # The prefix at every fraction is nested, deterministic and arm-independent.
        blocks = {}
        for ri, end in candidates:
            key = (self.runs[ri]["run_id"], (end-self.lag_tr)//int(block_tr))
            blocks.setdefault(key, []).append((ri, end))
        ordered_keys = sorted(blocks, key=lambda key: hashlib.sha256(
            f"{subset_seed}:{key[0]}:{key[1]}".encode()).hexdigest())
        ordered = [pair for key in ordered_keys for pair in blocks[key]]
        actual_fraction = float(label_fraction) if split == "train" else 1.0
        count = max(1, math.ceil(len(ordered) * actual_fraction))
        self.indices = sorted(ordered[:count])
        if not self.indices:
            raise ValueError("No valid semantic examples after verified timing alignment")
        ids = [f"{self.runs[ri]['run_id']}:{end-self.lag_tr}" for ri, end in self.indices]
        self.label_metadata = {
            "requested_fraction": actual_fraction, "selected_examples": len(self.indices),
            "eligible_examples": len(candidates), "realized_fraction": len(self.indices)/len(candidates),
            "subset_seed": subset_seed, "block_tr": block_tr,
            "indices_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "label_ids": ids,
        }

    def __getitem__(self, index):
        result = super().__getitem__(index)
        ri, end = self.indices[index]
        target_tr = end-self.lag_tr
        result["y"] = torch.from_numpy(np.array(self._array(ri, "target_path")[target_tr], dtype=np.float32, copy=True))
        result["target_tr"] = target_tr
        return result
