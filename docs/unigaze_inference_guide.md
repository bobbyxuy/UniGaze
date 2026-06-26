# UniGaze Model Download and Inference Guide

This guide describes the practical workflow for using this repository to run UniGaze gaze estimation on:

- a single image
- a directory of images
- a single video
- a directory of videos
- IR / grayscale inputs converted to 3-channel images

The recommended default model is `unigaze_h14_joint`.

## 1. Environment

On the WSL machine used for this project:

```bash
cd ~/eye_gaze/UniGaze
source ../.venv-unigaze/bin/activate
```

If you are using a fresh machine, install the package and dependencies first:

```bash
cd ~/eye_gaze/UniGaze
python -m pip install -U pip
python -m pip install -e .
```

For GPU inference, make sure PyTorch sees CUDA:

```bash
python - <<'PY'
import torch
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

## 2. Available model names

The upstream UniGaze package exposes these model names:

| Model name | Backbone | Training data | Recommended use |
|---|---|---|---|
| `unigaze_b16_joint` | UniGaze-B | Joint datasets | smaller / lighter baseline |
| `unigaze_l16_joint` | UniGaze-L | Joint datasets | middle-size model |
| `unigaze_h14_joint` | UniGaze-H | Joint datasets | **default strongest general model** |
| `unigaze_h14_cross_X` | UniGaze-H | ETH-XGaze | ETH-XGaze-like domain only |

For IR grayscale, DMS-like, webcam, or unknown input domains, use:

```text
unigaze_h14_joint
```

`unigaze_h14_cross_X` is not a larger model. It uses the same H14 backbone but is trained/adapted for the ETH-XGaze domain, so it is not automatically better for IR or grayscale inputs.

## 3. Model download

The simplest download path is to call `unigaze.load(...)`. On the first run, it downloads the model from HuggingFace into the local HF cache.

```bash
cd ~/eye_gaze/UniGaze
source ../.venv-unigaze/bin/activate
python - <<'PY'
import torch
import unigaze

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = unigaze.load('unigaze_h14_joint', device=device)
model.eval()
print('loaded unigaze_h14_joint on', device)
PY
```

On the WSL machine, the cached/downloaded model was observed at paths like:

```text
~/.cache/huggingface/hub/models--UniGaze--UniGaze-models/...
~/eye_gaze/UniGaze/unigaze/checkpoints/hf/unigaze_h14_joint.safetensors
```

Do **not** commit large `.safetensors` / `.pth` model weights into GitHub. Keep them in HuggingFace cache or a model artifact store.

## 4. Unified inference script

This repository adds:

```text
tools/unigaze_infer.py
```

It supports:

- image file input
- image directory input
- video file input
- video directory input
- optional grayscale/IR conversion using `--force-gray`
- annotated image/video outputs
- CSV prediction output

Run it from the repository root:

```bash
cd ~/eye_gaze/UniGaze
source ../.venv-unigaze/bin/activate
```

### Single image inference

```bash
python tools/unigaze_infer.py \
  -i /path/to/frame.jpg \
  -o outputs/frame \
  --model-name unigaze_h14_joint
```

Outputs:

```text
outputs/frame/images/frame_gaze.jpg
outputs/frame/predictions.csv
```

### Batch image folder inference

```bash
python tools/unigaze_infer.py \
  -i /path/to/image_folder \
  -o outputs/images \
  --model-name unigaze_h14_joint
```

Limit the number of images for a quick test:

```bash
python tools/unigaze_infer.py \
  -i /path/to/image_folder \
  -o outputs/images_test \
  --max-images 100
```

### IR / grayscale image inference

If your images are real IR grayscale, OpenCV may load them as 3-channel BGR anyway. To enforce grayscale-to-3-channel input, use:

```bash
python tools/unigaze_infer.py \
  -i /path/to/ir_images \
  -o outputs/ir_images \
  --model-name unigaze_h14_joint \
  --force-gray
```

`--force-gray` converts each image/frame to grayscale and replicates it into 3 channels before face alignment and UniGaze inference.

### Video inference

```bash
python tools/unigaze_infer.py \
  -i /path/to/video.mp4 \
  -o outputs/video \
  --model-name unigaze_h14_joint
```

Outputs:

```text
outputs/video/videos/video_gaze.mp4
outputs/video/predictions.csv
```

For faster video processing, skip frames:

```bash
python tools/unigaze_infer.py \
  -i /path/to/video.mp4 \
  -o outputs/video_fast \
  --skip-frames 2
```

This runs gaze inference every 2 frames and reuses the last annotated frame for skipped frames.

For CSV-only video inference:

```bash
python tools/unigaze_infer.py \
  -i /path/to/video.mp4 \
  -o outputs/video_csv \
  --no-video
```

### Save normalized crops

To inspect the 224x224 normalized face crops sent into UniGaze:

```bash
python tools/unigaze_infer.py \
  -i /path/to/images \
  -o outputs/debug \
  --write-normalized
```

This writes normalized crops to:

```text
outputs/debug/normalized/
```

## 5. Output CSV columns

`predictions.csv` contains one row per detected face per processed frame/image.

Important columns:

| Column | Meaning |
|---|---|
| `source` | original image/video path |
| `frame_index` | video frame index; `0` for still images |
| `face_index` | index of detected face in the frame |
| `pred_norm_pitch`, `pred_norm_yaw` | UniGaze prediction in normalized crop coordinates |
| `pred_camera_pitch`, `pred_camera_yaw` | denormalized prediction in camera/image coordinates |
| `gaze_start_x`, `gaze_start_y` | projected gaze arrow start point |
| `gaze_end_x`, `gaze_end_y` | projected gaze arrow end point |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | drawn face bounding box |

## 6. Original upstream video script

The upstream script is still available:

```bash
cd ~/eye_gaze/UniGaze/unigaze
source ../../.venv-unigaze/bin/activate
python predict_gaze_video.py \
  --model_name unigaze_h14_joint \
  -i /path/to/video_or_video_folder \
  -out /path/to/output
```

The added `tools/unigaze_infer.py` is more convenient when you need images, folders, videos, CSV outputs, and IR/grayscale handling in one command.

## 7. Practical recommendations

For general use:

```bash
python tools/unigaze_infer.py -i INPUT -o OUTPUT --model-name unigaze_h14_joint
```

For IR grayscale:

```bash
python tools/unigaze_infer.py -i INPUT -o OUTPUT --model-name unigaze_h14_joint --force-gray
```

For videos where speed matters:

```bash
python tools/unigaze_infer.py -i INPUT.mp4 -o OUTPUT --skip-frames 2
```

If face detection fails on IR frames, check:

- image exposure and contrast
- whether the full face is visible
- whether glasses/IR reflections obscure landmarks
- `--resize-factor`; try `--resize-factor 1.0` for difficult frames

## 8. RGB-to-gray robustness check

A local test on 1000 MPIIFaceGaze processed face patches showed that grayscale replicated to 3 channels only degraded `unigaze_h14_joint` by about `+0.12°` mean angular error versus original RGB/BGR input.

See:

```text
docs/unigaze_ir_gray_usage.md
tools/compare_unigaze_gray.py
```

This is only an RGB-to-gray simulation. Real IR can still differ because of glints, pupil appearance, glasses reflection, exposure, and noise.
