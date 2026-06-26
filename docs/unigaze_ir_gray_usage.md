# UniGaze H14 Joint for IR/Gray Gaze Estimation

This note summarizes how to use the local UniGaze setup for gaze estimation on WSL, and records a quick RGB-vs-gray robustness check for `unigaze_h14_joint`.

## Available local gaze models

On the WSL host, the useful gaze models found during inspection are:

| Model | Local path | Notes |
|---|---|---|
| `unigaze_h14_joint` | `~/eye_gaze/UniGaze/unigaze/checkpoints/hf/unigaze_h14_joint.safetensors` and HF cache | Strongest local UniGaze model; UniGaze-H backbone trained on joint datasets. Recommended default. |
| `unigaze_h14_cross_X` | loaded by `unigaze.load("unigaze_h14_cross_X")` if downloaded | Same H14 backbone, trained/adapted on ETH-XGaze. Not necessarily better for IR/gray inputs. |
| ResNet34 gaze | `~/gaze-estimation/weights/resnet34.pt`, `~/gaze-estimation/weights/resnet34_gaze.onnx` | Lighter deployment-oriented baseline. Lower expected accuracy/generalization than UniGaze-H. |

## Model choice

For IR grayscale input, prefer:

```text
unigaze_h14_joint
```

Reasoning:

- `h14_joint` is trained on joint datasets and is usually more robust across domains.
- `h14_cross_X` is ETH-XGaze-specific. It can be better when the input domain resembles ETH-XGaze, but IR grayscale driver-monitoring-style frames are a domain shift from ETH-XGaze RGB images.
- `h14_cross_X` is not a larger model than `h14_joint`; both are H14. The difference is training data/domain.

## Basic inference loading

From the UniGaze environment:

```bash
cd ~/eye_gaze/UniGaze/unigaze
source ../../.venv-unigaze/bin/activate
```

Python:

```python
import torch
import unigaze

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unigaze.load("unigaze_h14_joint", device=DEVICE)
model.eval()
```

## Handling IR grayscale frames

UniGaze expects 3-channel image input with ImageNet normalization. For a single-channel IR/grayscale frame, replicate it into 3 channels before applying the normal transform.

OpenCV example:

```python
import cv2
import numpy as np

# gray: H x W uint8 IR/grayscale image
gray = cv2.imread("frame_ir.png", cv2.IMREAD_GRAYSCALE)
image_rgb = np.stack([gray, gray, gray], axis=-1)
```

If the source frame is BGR and you want to simulate grayscale:

```python
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
```

Then use the same UniGaze transform used by the project:

```python
from datasets.helper.image_transform import wrap_transforms

transform = wrap_transforms("basic_imagenet", image_size=224)
x = transform(gray_rgb.astype("uint8")).unsqueeze(0).float().to(DEVICE)

with torch.no_grad():
    pred = model(x)["pred_gaze"]  # pitch/yaw-like 2D gaze output used by UniGaze
```

## RGB-vs-gray robustness test

A quick test was run on the local processed MPIIFaceGaze data:

- Model: `unigaze_h14_joint`
- Dataset: `~/eye_gaze/UniGaze/data/processed/mpiifacegaze_224`
- Split/sample: deterministic sample of 1000 face patches from `p10.h5` to `p14.h5`
- Original input: processed BGR face patch converted to RGB for the project transform
- Gray input: same image converted to grayscale and replicated to 3 channels
- Metric: angular error in degrees using `gazelib.gaze.gaze_utils.angular_error`

Results:

| Input | Mean angular error |
|---|---:|
| Original image | 5.06° |
| Gray replicated to 3 channels | 5.18° |
| Gray - original | +0.12° |

Prediction drift between original and gray input:

| Drift metric | Degrees |
|---|---:|
| Mean | 0.79° |
| Median | 0.71° |
| P90 | 1.35° |
| P95 | 1.61° |

Subject-level summary:

| Subject | Original | Gray 3-channel | Delta |
|---|---:|---:|---:|
| p10 | 6.46° | 6.76° | +0.30° |
| p11 | 3.48° | 3.67° | +0.19° |
| p12 | 3.36° | 3.94° | +0.59° |
| p13 | 4.88° | 4.75° | -0.13° |
| p14 | 7.22° | 6.90° | -0.31° |

Interpretation:

- UniGaze-H joint does not appear to depend strongly on color for this visible-light-to-grayscale test.
- The average degradation was small: about `+0.12°`.
- However, this is only an RGB-to-gray simulation. Real IR images may still differ because of pupil/iris appearance, eye glints, glasses reflection, exposure, noise, and illumination geometry.
- For real IR deployment, run the same comparison on real IR frames and, if needed, fine-tune or distill with IR data.

## Reproduce the comparison

Use the script in this repository:

```bash
cd ~/eye_gaze/UniGaze/unigaze
source ../../.venv-unigaze/bin/activate
N=1000 BATCH=32 python ../tools/compare_unigaze_gray.py
```

Notes:

- `BATCH=32` worked reliably on the tested WSL/CUDA environment.
- Larger runs at `N=1500`/`N=2000` were killed around 1280 processed samples, likely because of memory accumulation or WSL/GPU memory pressure.
- If this happens, reduce `N` or split the run into chunks.
