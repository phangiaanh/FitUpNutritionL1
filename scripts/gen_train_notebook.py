#!/usr/bin/env python3
"""Generate notebooks/train_l1_yolo11s.ipynb.

The notebook is built up from a list of cells defined inline below. Helper
functions used inside the notebook live in scripts/notebook_helpers.py and
are embedded verbatim into one of the cells (so the notebook stays
self-contained when run in Colab).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "notebooks" / "train_l1_yolo11s.ipynb"
HELPERS_PATH = REPO_ROOT / "scripts" / "notebook_helpers.py"


def jl(text: str) -> list[str]:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": jl(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": jl(text),
        "outputs": [],
        "execution_count": None,
    }


def build_cells() -> list[dict]:
    helpers_source = HELPERS_PATH.read_text()
    cells: list[dict] = []

    # Cell 1: Markdown header
    cells.append(markdown(
        r"""# L1 Food detector - YOLO11s

Trains an **Ultralytics YOLO11s** object detector on 8 L1 food categories.
Dataset loads from **[Hugging Face `WatermelonAnh/FoodClassifierL1`](https://huggingface.co/datasets/WatermelonAnh/FoodClassifierL1)**
as two tar archives (`images.tar`, `labels.tar`) in YOLO label format.

### Classes (fixed order, 8)
`noodle_dish`, `rice_dish`, `soup_stew`, `grilled_fried`, `banh_bread`, `beverage`, `fruit`, `dessert_snack`

### Deployment target
Final artifact is a quantized `.tflite` file for on-device mobile inference.
Both INT8 and FP16 TFLite files are exported at the end of the notebook.

Checkpoints and exports persist to **Google Drive** so a Colab disconnect
never destroys progress."""
    ))

    # Cell 2: pip install
    cells.append(code(
        "%%capture\n"
        "%pip install -q --upgrade ultralytics huggingface_hub pyyaml pillow tqdm"
    ))

    # Cell 3: Imports + GPU check
    cells.append(code(
        r"""from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import login, snapshot_download

if not torch.cuda.is_available():
    raise SystemExit(
        "GPU required - Runtime -> Change runtime type -> GPU (A100 recommended)."
    )

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    "GB",
)"""
    ))

    # Cell 4: Drive mount + HF login
    cells.append(code(
        r"""try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ImportError:
    print("Outside Colab - ensure DRIVE_ROOT exists on the host filesystem.")

TOKEN = os.environ.get("HF_TOKEN", "").strip()
if TOKEN:
    login(token=TOKEN)
else:
    login()
print("Hugging Face login OK.")"""
    ))

    # Cell 5: Config
    cells.append(code(
        r"""DRIVE_ROOT  = "/content/drive/MyDrive/FitUpNutritionL1"
RUNS_DIR    = os.path.join(DRIVE_ROOT, "runs")
EXPORTS_DIR = os.path.join(DRIVE_ROOT, "exports")

HF_DATASET_REPO = "WatermelonAnh/FoodClassifierL1"
HF_CACHE        = "/content/hf_cache"
DATA_DIR        = "/content/l1_dataset"
DATA_YAML       = os.path.join(DATA_DIR, "data.yaml")

RUN_NAME = "l1_yolo11s"

L1_CLASSES = [
    "noodle_dish",
    "rice_dish",
    "soup_stew",
    "grilled_fried",
    "banh_bread",
    "beverage",
    "fruit",
    "dessert_snack",
]

# Training hyperparameters (A100-tuned)
IMG_SIZE  = 640
EPOCHS    = 100
BATCH     = 64
PATIENCE  = 30
WORKERS   = 8
CACHE     = "ram"   # falls back to "disk" if RAM is insufficient
AMP       = True

FORCE_REDOWNLOAD = False

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
print("Config OK. Run name:", RUN_NAME)"""
    ))

    # Cell 6: Helpers (embedded notebook_helpers.py)
    cells.append(markdown(
        "### Helpers\n\n"
        "The next cell is the verbatim contents of `scripts/notebook_helpers.py` "
        "from the repo. Re-run this cell after pulling helper updates."
    ))
    cells.append(code(helpers_source.rstrip()))

    # Cell: dataset download + extract + data.yaml
    cells.append(markdown("### Dataset: download tars from HF, extract, write data.yaml"))
    cells.append(code(
        r"""snapshot_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    local_dir=HF_CACHE,
)

extract_dataset_tars(HF_CACHE, DATA_DIR, force=FORCE_REDOWNLOAD)
yaml_path = write_data_yaml(DATA_DIR, L1_CLASSES)
print("data.yaml ->", yaml_path)

for split in ("train", "val", "test"):
    imgs = list((Path(DATA_DIR) / "images" / split).glob("*"))
    print(f"  {split}: {len(imgs)} images")"""
    ))

    # Cell: pre-flight label validation
    cells.append(markdown(
        "### Label validation\n\n"
        "Fails loudly if any `.txt` is malformed or if there are orphan "
        "images/labels. Run this **before** training."
    ))
    cells.append(code("validate_yolo_labels(DATA_DIR, L1_CLASSES)"))

    # Cell: training (with resume detection)
    cells.append(markdown(
        "### Train\n\n"
        "Single-stage fine-tune from COCO-pretrained `yolo11s.pt`. "
        "If a previous run was interrupted, the last checkpoint on Drive is "
        "resumed automatically."
    ))
    cells.append(code(
        r"""from ultralytics import YOLO

LAST_PT = os.path.join(RUNS_DIR, RUN_NAME, "weights", "last.pt")
if os.path.isfile(LAST_PT):
    print(f"Resuming from {LAST_PT}")
    model = YOLO(LAST_PT)
    resume=True
else:
    print("Starting fresh from yolo11s.pt")
    model = YOLO("yolo11s.pt")
    resume=False

model.train(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    patience=PATIENCE,
    workers=WORKERS,
    cache=CACHE,
    amp=AMP,
    device=0,
    project=RUNS_DIR,
    name=RUN_NAME,
    exist_ok=True,
    resume=resume,
)"""
    ))

    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": build_cells(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
