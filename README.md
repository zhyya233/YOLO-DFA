# YOLO-DFA

This repository contains the code released with the manuscript:

**YOLO-DFA: Dynamic Feature-Aware Fusion for Small Object Detection**

YOLO-DFA is built on the Ultralytics YOLO framework and introduces several custom modules for small-object detection: `DynamicConv`, `C2f_Bifocal`, `DK_FMM`, an efficient iRMB-based GFPN neck with `iRMB_Zoom`, and the pre-head `SSEM` calibration module. The model configuration keeps the anchor-free YOLO detection head and adds a four-scale P2-P5 prediction structure.

## Repository Structure

```text
YOLO-DFA-Code/
  Model/
    YOLO-DFA.yaml        # YOLO-DFA model configuration
    YOLOv8s.yaml         # YOLOv8s baseline configuration
  Modules/
    DynamicConv.py       # CondConv-based dynamic convolution and C2f_DynamicConv
    C2f_Bifocal.py       # Bifocal C2f feature transformation block
    DK_FMM.py            # Deformable Kernel Focal Modulation module
    GFPN.py              # GFPN-related fusion components
    iRMB.py              # iRMB baseline components
    iRMB_Zoom.py         # iRMB-Zoom local-detail refinement module
    SSEM.py              # Stabilized Saliency Enhancement Module
  yolo/                  # Ultralytics YOLO engine files used by the project
  requirements.txt       # Environment export from the experimental setup
```

## Environment

The experiments were conducted with PyTorch and Ultralytics YOLO. A Python 3.10 environment is recommended.

Core packages:

```text
torch==2.4.1
torchvision==0.19.1
ultralytics==8.3.203
timm==1.0.20
einops==0.8.1
thop==0.1.1.post2209072238
opencv-python
numpy
pandas
matplotlib
PyYAML
tqdm
scipy
```

Example installation:

```bash
conda create -n yolo-dfa python=3.10 -y
conda activate yolo-dfa
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The provided `requirements.txt` was exported from the main experimental environment. If `pip` reports platform-specific local paths such as `file:///C:/...`, install the core packages listed above manually or clean the requirement file before installation.

## Integrating the Custom Modules

This release is based on Ultralytics YOLO. The files in `Modules/` should be used together with an Ultralytics source tree in which the custom modules are registered in the model parser.

For a fresh Ultralytics source tree, copy the module files into the `ultralytics/nn/` directory:

```bash
cp Modules/*.py /path/to/ultralytics/ultralytics/nn/
```

Then make sure the following modules are imported and registered in the Ultralytics model parser:

```python
from ultralytics.nn.DynamicConv import DynamicConv, C2f_DynamicConv
from ultralytics.nn.C2f_Bifocal import C2f_Bifocal
from ultralytics.nn.DK_FMM import DK_FMM
from ultralytics.nn.iRMB_Zoom import iRMB_Zoom
from ultralytics.nn.SSEM import SSEM
```

The model configuration `Model/YOLO-DFA.yaml` uses the module names `DynamicConv`, `C2f_DynamicConv`, `C2f_Bifocal`, `DK_FMM`, `iRMB_Zoom`, and `SSEM`. Therefore, these names must be visible to the Ultralytics YAML parser before model construction.

## Quick Sanity Check

After the custom modules are registered, verify that the model can be constructed before starting a full training run:

```bash
yolo detect train model=Model/YOLO-DFA.yaml data=coco128.yaml imgsz=640 epochs=1 batch=1 device=0
```

This command is only a construction and smoke-test check. It is not intended to reproduce the paper results.

You can also test the standalone modules:

```bash
python -m py_compile Modules/*.py
```

## Datasets

The paper evaluates YOLO-DFA on four public datasets:

```text
VisDrone2019-DET
TT100K
NWPU-VHR-10
RSOD
```

Datasets are not redistributed in this repository. Please download them from their official sources and convert the annotations to the standard YOLO detection format:

```text
dataset_root/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

A dataset YAML file should follow the usual Ultralytics format:

```yaml
path: /absolute/path/to/dataset_root
train: images/train
val: images/val
test: images/test

names:
  0: class_0
  1: class_1
```

The split protocol used in the manuscript is:

```text
VisDrone2019-DET: train partition for training, validation partition for local evaluation.
TT100K: 6107 images for training and 3073 images for testing, using 45 selected categories.
NWPU-VHR-10: 390 training images and 260 test images from the annotated subset; 150 background images are included in training.
RSOD: 586 training images, 195 validation images, and 195 test images.
```

## Training

The main YOLO-DFA model can be trained from scratch with:

```bash
yolo detect train \
  model=Model/YOLO-DFA.yaml \
  data=/path/to/dataset.yaml \
  imgsz=640 \
  epochs=300 \
  batch=8 \
  optimizer=AdamW \
  lr0=0.0002 \
  lrf=0.01 \
  weight_decay=0.0005 \
  cos_lr=True \
  warmup_epochs=5 \
  mosaic=1.0 \
  close_mosaic=10 \
  amp=True \
  workers=8 \
  seed=0 \
  pretrained=False \
  device=0
```

For the three-run VisDrone comparison reported in the manuscript, use seeds `0`, `42`, and `123` under the same training protocol.

The YOLOv8s baseline configuration is provided at:

```text
Model/YOLOv8s.yaml
```

## Validation

Evaluate a trained checkpoint with:

```bash
yolo detect val \
  model=/path/to/best.pt \
  data=/path/to/dataset.yaml \
  imgsz=640 \
  batch=1 \
  conf=0.001 \
  iou=0.7 \
  device=0
```

When a validation split is available, the best checkpoint is selected according to validation mAP@0.5:0.95. For datasets or protocols without a validation split, the final checkpoint after 300 epochs is evaluated on the test split, so that test annotations are not used for model selection.

## Notes for Reproducibility

- All internally reproduced models in the manuscript are trained from scratch without external pretrained weights.
- Input images are resized to `640 x 640` using the Ultralytics letterbox pipeline.
- FLOPs and parameter counts are computed at `640 x 640` using the Ultralytics profiling interface with THOP.
- Reported latency in the manuscript was measured in FP32 mode with batch size 1 after warm-up iterations.
- Small numerical differences may occur across CUDA, PyTorch, and GPU versions.

## Troubleshooting

If the YAML parser reports that `SSEM`, `DynamicConv`, `C2f_Bifocal`, `DK_FMM`, or `iRMB_Zoom` is not defined, the custom modules have not been registered in the active Ultralytics source tree.

If `pip install -r requirements.txt` fails because of `file:///C:/...` entries, remove those platform-specific entries or install the core packages listed in the Environment section manually.

If `torchvision.ops.DeformConv2d` is unavailable, check that the installed `torch` and `torchvision` versions are compatible with each other and with the CUDA version on your machine.

## License

This project is derived from the Ultralytics YOLO framework. Please follow the license terms of the upstream Ultralytics project and the licenses of the datasets used in your experiments.
