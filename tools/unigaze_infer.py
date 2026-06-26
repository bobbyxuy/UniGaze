#!/usr/bin/env python3
"""UniGaze inference helper for images, image folders, and videos.

Examples:
  # First run downloads weights from HuggingFace cache if needed.
  python tools/unigaze_infer.py -i samples/frame.jpg -o outputs --model-name unigaze_h14_joint
  python tools/unigaze_infer.py -i samples/images -o outputs/images --force-gray
  python tools/unigaze_infer.py -i samples/video.mp4 -o outputs/video --skip-frames 2

This script reuses the normalization path from unigaze/predict_gaze_video.py,
adds batch image-folder support, writes annotated outputs, and records CSV
predictions.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from glob import glob
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIGAZE_DIR = REPO_ROOT / "unigaze"
os.chdir(UNIGAZE_DIR)
sys.path.insert(0, str(UNIGAZE_DIR))

from datasets.helper.image_transform import wrap_transforms  # noqa: E402
from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw  # noqa: E402
from gazelib.gaze.normalize import estimateHeadPose, normalize  # noqa: E402
from gazelib.label_transform import get_face_center_by_nose  # noqa: E402
import face_alignment  # noqa: E402
import unigaze  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ARROW_COLORS = [(47, 255, 173), (255, 173, 47), (173, 47, 255), (47, 173, 255)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UniGaze on images, image folders, or videos.")
    parser.add_argument("-i", "--input", required=True, help="Image file, image directory, video file, or video directory.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("--model-name", default="unigaze_h14_joint", help="UniGaze model name, e.g. unigaze_h14_joint or unigaze_h14_cross_X.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--force-gray", action="store_true", help="Convert every frame/image to grayscale and replicate to 3 channels before inference. Useful for simulating IR grayscale input.")
    parser.add_argument("--skip-frames", type=int, default=1, help="Video frame stride. 1 = process every frame; 2 = every other frame.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images from a directory. 0 = no limit.")
    parser.add_argument("--write-normalized", action="store_true", help="Save normalized 224x224 face crops with gaze arrows.")
    parser.add_argument("--no-video", action="store_true", help="For video input, write CSV only and skip annotated video output.")
    parser.add_argument("--resize-factor", type=float, default=0.5, help="Resize factor for face landmark detection. Lower is faster; 1 keeps original size.")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_gray3_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def set_dummy_camera_model(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    focal_length = w * 4
    center = (w // 2, h // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    camera_distortion = np.zeros((1, 5))
    return camera_matrix, camera_distortion


def denormalize_predicted_gaze(gaze_pitchyaw: np.ndarray, r_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_gaze = pitchyaw_to_vector(gaze_pitchyaw.reshape(1, 2)).reshape(3, 1)
    pred_gaze = np.matmul(r_inv, pred_gaze.reshape(3, 1))
    pred_gaze = pred_gaze / np.linalg.norm(pred_gaze)
    pred_pitchyaw = vector_to_pitchyaw(pred_gaze.reshape(1, 3))
    return pred_gaze, pred_pitchyaw


def draw_gaze_on_crop(image_bgr: np.ndarray, pitchyaw: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    if out.ndim == 2 or out.shape[2] == 1:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    length = w / 2.0
    start = (int(w / 2.0), int(h / 2.0))
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    end = (int(start[0] + dx), int(start[1] + dy))
    cv2.arrowedLine(out, start, end, color, 3, cv2.LINE_AA, tipLength=0.3)
    return out


def iter_inputs(path: Path) -> tuple[str, list[Path]]:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            return "images", [path]
        if suffix in VIDEO_EXTS:
            return "videos", [path]
        raise ValueError(f"Unsupported input file extension: {path.suffix}")
    if not path.is_dir():
        raise FileNotFoundError(path)

    images = sorted([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    videos = sorted([p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS])
    if images and videos:
        raise ValueError("Input directory contains both images and videos; pass a more specific directory.")
    if images:
        return "images", images
    if videos:
        return "videos", videos
    raise ValueError(f"No supported images or videos found in {path}")


def init_face_alignment():
    try:
        return face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    except AttributeError:
        return face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


def process_frame(
    image_bgr: np.ndarray,
    *,
    model,
    transform,
    face_aligner,
    device: torch.device,
    resize_factor: float,
    force_gray: bool,
    source: str,
    frame_index: int,
    normalized_dir: Path | None,
) -> tuple[np.ndarray, list[dict]]:
    if force_gray:
        image_bgr = make_gray3_bgr(image_bgr)

    image_out = image_bgr.copy()
    detect_image = image_bgr
    if resize_factor < 1.0:
        detect_image = cv2.resize(image_bgr, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    detect_rgb = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
    preds = face_aligner.get_landmarks(detect_rgb)
    records: list[dict] = []
    if preds is None:
        return image_out, records

    focal_norm = 960
    distance_norm = 600
    roi_size = (224, 224)
    face_model_load = np.loadtxt("data/face_model.txt")
    face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]
    face_pts = face_model.reshape(6, 1, 3)

    for face_idx, landmarks_in_original in enumerate(preds):
        color = ARROW_COLORS[face_idx % len(ARROW_COLORS)]
        landmarks_in_original = landmarks_in_original / resize_factor
        x_min = int(landmarks_in_original[:, 0].min())
        x_max = int(landmarks_in_original[:, 0].max())
        y_min = int(landmarks_in_original[:, 1].min())
        y_max = int(landmarks_in_original[:, 1].max())
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        if bbox_width <= 2 or bbox_height <= 2:
            continue
        bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        draw_scale = 1.2
        x_min_draw = max(0, bbox_center[0] - int(bbox_width * draw_scale // 2))
        x_max_draw = min(image_bgr.shape[1], bbox_center[0] + int(bbox_width * draw_scale // 2))
        y_min_draw = max(0, bbox_center[1] - int(bbox_height * draw_scale // 2))
        y_max_draw = min(image_bgr.shape[0], bbox_center[1] + int(bbox_height * draw_scale // 2))

        crop_scale = 2.0
        crop_x_min = max(0, bbox_center[0] - int(bbox_width * crop_scale // 2))
        crop_x_max = min(image_bgr.shape[1], bbox_center[0] + int(bbox_width * crop_scale // 2))
        crop_y_min = max(0, bbox_center[1] - int(bbox_height * crop_scale // 2))
        crop_y_max = min(image_bgr.shape[0], bbox_center[1] + int(bbox_height * crop_scale // 2))
        crop = image_bgr[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        if crop.size == 0:
            continue
        landmarks = landmarks_in_original - np.array([crop_x_min, crop_y_min])

        camera_matrix, camera_distortion = set_dummy_camera_model(crop)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :].astype(float).reshape(6, 1, 2)
        try:
            hr, ht = estimateHeadPose(landmarks_sub, face_pts, camera_matrix, camera_distortion)
            h_r = cv2.Rodrigues(hr)[0]
            face_center_camera, _ = get_face_center_by_nose(hR=h_r, ht=ht, face_model_load=face_model_load)
            img_normalized, r_norm, h_r_norm, _gaze_normalized, _landmarks_normalized, _ = normalize(
                crop,
                landmarks,
                focal_norm,
                distance_norm,
                roi_size,
                face_center_camera,
                hr,
                ht,
                camera_matrix,
                gc=None,
            )
        except Exception:
            continue

        hr_norm = np.array([np.arcsin(h_r_norm[1, 2]), np.arctan2(h_r_norm[0, 2], h_r_norm[2, 2])])
        if np.linalg.norm(hr_norm) > 80 * np.pi / 180:
            continue

        input_rgb = img_normalized[:, :, [2, 1, 0]]
        input_var = transform(input_rgb).float().to(device).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_var)["pred_gaze"][0].detach().cpu().numpy()

        if normalized_dir is not None:
            norm_vis = draw_gaze_on_crop(img_normalized, pred, color)
            normalized_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(normalized_dir / f"{Path(source).stem}_frame{frame_index:06d}_face{face_idx}.jpg"), norm_vis)

        r_inv = np.linalg.inv(r_norm)
        pred_vec, pred_pitchyaw_camera = denormalize_predicted_gaze(pred, r_inv)
        vec_length = pred_vec * -112 * 1.5
        gaze_ray = np.concatenate((face_center_camera.reshape(1, 3), (face_center_camera + vec_length).reshape(1, 3)), axis=0)
        result = cv2.projectPoints(
            gaze_ray,
            np.array([0, 0, 0]).reshape(3, 1).astype(float),
            np.array([0, 0, 0]).reshape(3, 1).astype(float),
            camera_matrix,
            camera_distortion,
        )[0].reshape(2, 2)
        result += np.array([crop_x_min, crop_y_min])
        start_pt = (int(result[0][0]), int(result[0][1]))
        end_pt = (int(result[1][0]), int(result[1][1]))

        cv2.rectangle(image_out, (x_min_draw, y_min_draw), (x_max_draw, y_max_draw), (0, 0, 240), 2)
        cv2.arrowedLine(image_out, start_pt, end_pt, color, 3, cv2.LINE_AA, tipLength=0.2)

        records.append(
            {
                "source": source,
                "frame_index": frame_index,
                "face_index": face_idx,
                "pred_norm_pitch": float(pred[0]),
                "pred_norm_yaw": float(pred[1]),
                "pred_camera_pitch": float(pred_pitchyaw_camera.reshape(-1)[0]),
                "pred_camera_yaw": float(pred_pitchyaw_camera.reshape(-1)[1]),
                "gaze_start_x": start_pt[0],
                "gaze_start_y": start_pt[1],
                "gaze_end_x": end_pt[0],
                "gaze_end_y": end_pt[1],
                "bbox_x1": x_min_draw,
                "bbox_y1": y_min_draw,
                "bbox_x2": x_max_draw,
                "bbox_y2": y_max_draw,
            }
        )
    return image_out, records


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "frame_index",
        "face_index",
        "pred_norm_pitch",
        "pred_norm_yaw",
        "pred_camera_pitch",
        "pred_camera_yaw",
        "gaze_start_x",
        "gaze_start_y",
        "gaze_end_x",
        "gaze_end_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_images(paths: list[Path], args, model, transform, face_aligner, device: torch.device) -> None:
    if args.max_images > 0:
        paths = paths[: args.max_images]
    out_dir = Path(args.output)
    image_out_dir = out_dir / "images"
    norm_dir = out_dir / "normalized" if args.write_normalized else None
    image_out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for path in tqdm(paths, desc="images"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        annotated, rows = process_frame(
            image,
            model=model,
            transform=transform,
            face_aligner=face_aligner,
            device=device,
            resize_factor=args.resize_factor,
            force_gray=args.force_gray,
            source=str(path),
            frame_index=0,
            normalized_dir=norm_dir,
        )
        out_path = image_out_dir / f"{path.stem}_gaze.jpg"
        cv2.imwrite(str(out_path), annotated)
        all_rows.extend(rows)
    write_csv(out_dir / "predictions.csv", all_rows)
    print(f"Saved {len(paths)} annotated images to {image_out_dir}")
    print(f"Saved {len(all_rows)} face predictions to {out_dir / 'predictions.csv'}")


def process_videos(paths: list[Path], args, model, transform, face_aligner, device: torch.device) -> None:
    out_dir = Path(args.output)
    video_out_dir = out_dir / "videos"
    norm_dir = out_dir / "normalized" if args.write_normalized else None
    video_out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Warning: cannot open video {path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if not args.no_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_out_dir / f"{path.stem}_gaze.mp4"), fourcc, fps, (width, height))
        frame_index = 0
        pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=path.name)
        last_annotated = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % max(1, args.skip_frames) == 0:
                annotated, rows = process_frame(
                    frame,
                    model=model,
                    transform=transform,
                    face_aligner=face_aligner,
                    device=device,
                    resize_factor=args.resize_factor,
                    force_gray=args.force_gray,
                    source=str(path),
                    frame_index=frame_index,
                    normalized_dir=norm_dir,
                )
                last_annotated = annotated
                all_rows.extend(rows)
            else:
                annotated = last_annotated if last_annotated is not None else frame
            if writer is not None:
                writer.write(annotated)
            frame_index += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
    write_csv(out_dir / "predictions.csv", all_rows)
    print(f"Saved video outputs to {video_out_dir}")
    print(f"Saved {len(all_rows)} face predictions to {out_dir / 'predictions.csv'}")


def main() -> None:
    args = parse_args()
    input_type, paths = iter_inputs(Path(args.input))
    device = select_device(args.device)
    print(f"Loading model {args.model_name} on {device} ...")
    model = unigaze.load(args.model_name, device=device)
    model.eval()
    transform = wrap_transforms("basic_imagenet", image_size=224)
    face_aligner = init_face_alignment()
    if input_type == "images":
        process_images(paths, args, model, transform, face_aligner, device)
    else:
        process_videos(paths, args, model, transform, face_aligner, device)


if __name__ == "__main__":
    main()
