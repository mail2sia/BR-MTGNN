import csv

import torch
import torch.nn as nn


def dump_ckpt_vs_model_csv(ckpt_path: str, model: torch.nn.Module, out_csv: str) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print("[dump_ckpt_vs_model_csv] load fail:", e)
        return

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    elif isinstance(ckpt, nn.Module):
        sd = ckpt.state_dict()
    else:
        print("[dump_ckpt_vs_model_csv] unsupported ckpt format")
        return
    model_sd = model.state_dict()
    rows = []
    ck_keys = set(sd.keys())
    model_keys = set(model_sd.keys())
    for k in sorted(ck_keys | model_keys):
        if k in ck_keys and k in model_keys:
            status = "match" if tuple(sd[k].shape) == tuple(model_sd[k].shape) else "shape_mismatch"
        elif k in ck_keys:
            status = "unexpected"
        else:
            status = "missing"
        ck_shape = tuple(sd[k].shape) if k in ck_keys else ""
        model_shape = tuple(model_sd[k].shape) if k in model_keys else ""
        rows.append((k, ck_shape, model_shape, status))
    try:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "ckpt_shape", "model_shape", "status"])
            for r in rows:
                w.writerow(r)
        print(f"[dump_ckpt_vs_model_csv] wrote {len(rows)} rows to {out_csv}")
    except Exception as e:
        print("[dump_ckpt_vs_model_csv] write fail:", e)


def _infer_ckpt_num_nodes(path: str) -> int | None:
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt.get("state_dict", {})
        elif isinstance(ckpt, dict):
            sd = ckpt
        elif isinstance(ckpt, nn.Module):
            sd = ckpt.state_dict()
        else:
            return None
        if isinstance(sd, dict) and "gc.emb1.weight" in sd:
            return int(sd["gc.emb1.weight"].shape[0])
    except Exception:
        return None
    return None


def _infer_scaler_num_nodes(path: str) -> int | None:
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "mean" in obj:
            return int(obj["mean"].numel())
    except Exception:
        return None
    return None
