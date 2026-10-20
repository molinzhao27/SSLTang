"""Read-only held-out story evaluation of continuous semantic predictions."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..models import FmriEncoder
from ..semantic.model import SemanticRegressor
from ..windows import SemanticWindows
from ..runtime import file_sha256, save_json
from .metrics import evaluate_embeddings


def evaluate_story(checkpoint_path, manifest_path=None, split="test", output_dir=None,
                   device="cuda", batch_size=256, num_workers=4, exclusion_tr=10):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "semantic":
        raise ValueError("Evaluation requires a supervised semantic checkpoint")
    config = checkpoint["config"]
    manifest_path = manifest_path or config["data"]["manifest"]
    dataset = SemanticWindows(manifest_path, split, config["data"]["window"],
                              config["data"]["stride"], lag_tr=config["semantic"]["lag_tr"])
    if split not in {"val", "test"}:
        raise ValueError("Use evaluate_transfer.py for ds004510")
    if file_sha256(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("Story evaluation must use the original training/split manifest")
    model = SemanticRegressor(FmriEncoder(checkpoint["encoder_config"]), dataset.target_dim,
                               config["semantic"].get("pooling", "end"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    mean = checkpoint["target_mean"].numpy()
    std = checkpoint["target_std"].numpy()
    predictions, targets, run_ids, times = [], [], [], []
    precision = config["training"].get("precision", "bf16")
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=device.startswith("cuda")):
            with torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                                enabled=device.startswith("cuda") and precision != "fp32"):
                prediction = model(batch["x"].to(device, non_blocking=True))
            predictions.append(prediction.float().cpu().numpy()*std+mean)
            targets.append(batch["y"].numpy())
            run_ids.extend(batch["run_id"])
            times.extend(batch["target_tr"].tolist())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    common = np.broadcast_to(mean, target.shape)
    kwargs = dict(run_ids=np.array(run_ids), times=np.array(times), exclusion_tr=exclusion_tr)
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": file_sha256(manifest_path), "split": split,
        "init": config["semantic"]["init"], "seed": config["training"]["seed"],
        "label_fraction": config["semantic"]["label_fraction"],
        "subject": dataset.runs[0]["subject"],
        "raw": evaluate_embeddings(prediction, target, **kwargs),
        "centered": evaluate_embeddings(prediction-mean, target-mean, **kwargs),
        "constant_train_mean": evaluate_embeddings(common, target, **kwargs),
        "normalized_mse": float(np.mean(((prediction-target)/std)**2)),
        "constant_normalized_mse": float(np.mean(((common-target)/std)**2)),
        "sessions": {r["run_id"]: r.get("session") for r in dataset.runs},
        "metric_note": "Centered cosine/retrieval subtract the selected training-target mean only. "
                       "Raw GPT2 cosine is anisotropic; use correlation, retrieval and the constant baseline together.",
    }
    out = Path(output_dir) if output_dir else Path(checkpoint_path).parent / f"evaluation_{split}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"predictions.npz", prediction=prediction, target=target,
                        run_ids=np.asarray(run_ids), target_tr=np.asarray(times), train_target_mean=mean)
    save_json(report, out/"metrics.json")
    print(json.dumps({k: report[k] for k in ["init", "seed", "label_fraction", "normalized_mse"]}), flush=True)
    return report
