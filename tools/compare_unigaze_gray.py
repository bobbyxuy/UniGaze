#!/usr/bin/env python3
"""Compare UniGaze predictions on original vs grayscale-replicated MPIIFaceGaze face patches.

Run from the UniGaze package directory:

    cd ~/eye_gaze/UniGaze/unigaze
    source ../../.venv-unigaze/bin/activate
    N=1000 BATCH=32 python ../tools/compare_unigaze_gray.py
"""

import json
import os
import random
import sys
import time

import cv2
import h5py
import numpy as np
import torch

ROOT = os.environ.get("UNIGAZE_ROOT", "/home/bobby/eye_gaze/UniGaze/unigaze")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datasets.helper.image_transform import wrap_transforms  # noqa: E402
from gazelib.gaze.gaze_utils import angular_error  # noqa: E402

DATA_DIR = os.environ.get("MPII_H5_DIR", "/home/bobby/eye_gaze/UniGaze/data/processed/mpiifacegaze_224")
KEYS = os.environ.get("MPII_KEYS", "p10.h5,p11.h5,p12.h5,p13.h5,p14.h5").split(",")
N = int(os.environ.get("N", "1000"))
SEED = int(os.environ.get("SEED", "42"))
BATCH = int(os.environ.get("BATCH", "32"))
MODEL_NAME = os.environ.get("MODEL_NAME", "unigaze_h14_joint")


def to_gray3_bgr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    items = []
    for key in KEYS:
        path = os.path.join(DATA_DIR, key)
        with h5py.File(path, "r") as f:
            count = f["face_patch"].shape[0]
        items += [(key, i) for i in range(count)]
    random.shuffle(items)
    items = items[: min(N, len(items))]

    transform = wrap_transforms("basic_imagenet", image_size=224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "event": "setup",
                "device": str(device),
                "model": MODEL_NAME,
                "samples": len(items),
                "batch": BATCH,
                "keys": KEYS,
                "data_dir": DATA_DIR,
            }
        ),
        flush=True,
    )

    import unigaze  # noqa: E402

    model = unigaze.load(MODEL_NAME, device=device)
    model.eval()

    all_gt = []
    all_pred_orig = []
    all_pred_gray = []
    subjects = []
    started = time.time()

    files = {key: h5py.File(os.path.join(DATA_DIR, key), "r", swmr=True) for key in KEYS}
    try:
        for start in range(0, len(items), BATCH):
            batch = items[start : start + BATCH]
            xs_orig, xs_gray, gts = [], [], []
            for key, index in batch:
                h5 = files[key]
                img_bgr = h5["face_patch"][index]
                gt = h5["face_gaze"][index].astype(float)

                # The MPII dataset config uses color_type=bgr; its preprocessing converts BGR to RGB.
                img_rgb = img_bgr[..., ::-1]
                gray_bgr = to_gray3_bgr(img_bgr)
                gray_rgb = gray_bgr[..., ::-1]

                xs_orig.append(transform(img_rgb.astype(np.uint8)))
                xs_gray.append(transform(gray_rgb.astype(np.uint8)))
                gts.append(gt)
                subjects.append(int(key[1:3]))

            x_orig = torch.stack(xs_orig).float().to(device)
            x_gray = torch.stack(xs_gray).float().to(device)
            with torch.no_grad():
                pred_orig = model(x_orig)["pred_gaze"].detach().cpu().numpy()
                pred_gray = model(x_gray)["pred_gaze"].detach().cpu().numpy()

            all_pred_orig.append(pred_orig)
            all_pred_gray.append(pred_gray)
            all_gt.append(np.asarray(gts))

            if (start // BATCH + 1) % 5 == 0:
                print(json.dumps({"event": "progress", "done": min(start + BATCH, len(items)), "total": len(items)}), flush=True)
    finally:
        for h5 in files.values():
            h5.close()

    pred_orig = np.concatenate(all_pred_orig, 0)
    pred_gray = np.concatenate(all_pred_gray, 0)
    gt = np.concatenate(all_gt, 0)
    subjects_np = np.asarray(subjects)

    err_orig = angular_error(pred_orig, gt)
    err_gray = angular_error(pred_gray, gt)
    drift = angular_error(pred_orig, pred_gray)

    by_subject = {}
    for subject in sorted(set(subjects_np.tolist())):
        mask = subjects_np == subject
        by_subject[f"p{subject:02d}"] = {
            "n": int(mask.sum()),
            "orig_mean": float(err_orig[mask].mean()),
            "gray_mean": float(err_gray[mask].mean()),
            "delta_mean": float((err_gray[mask] - err_orig[mask]).mean()),
            "drift_mean": float(drift[mask].mean()),
        }

    result = {
        "event": "result",
        "device": str(device),
        "model": MODEL_NAME,
        "dataset": "MPIIFaceGaze processed face_patch sample",
        "n": int(len(gt)),
        "seconds": round(time.time() - started, 2),
        "orig_error_deg": stats(err_orig),
        "gray3_error_deg": stats(err_gray),
        "gray_minus_orig_error_deg": stats(err_gray - err_orig),
        "prediction_drift_orig_vs_gray_deg": stats(drift),
        "gray_better_fraction": float(np.mean(err_gray < err_orig)),
        "by_subject": by_subject,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
