#!/usr/bin/env python3
"""Generate notebooks/train_l1_efficientnetb4.ipynb."""

from __future__ import annotations

import json
from pathlib import Path


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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "train_l1_efficientnetb4.ipynb"

CELLS = [
    markdown(
        r"""# L1 Food classifier - EfficientNet-B4 (multi-label)

Trains **EfficientNet-B4** with a sigmoid/BCE-with-logits head for **multi-label** L1 scenes.
Dataset loads from **[Hugging Face `WatermelonAnh/FoodClassifierL1`](https://huggingface.co/datasets/WatermelonAnh/FoodClassifierL1)**.

### Expected dataset layout under the HF repo snapshot

```
 FoodClassifierL1/
   train/<class>/images...
   val/<class>/...
   test/<class>/...
```

**Classes (fixed order, 8):**
`noodle_dish`, `rice_dish`, `soup_stew`, `grilled_fried`, `banh_bread`, `beverage`, `fruit`, `dessert_snack`

Checkpoints persist to **Google Drive** (`CKPT_DIR`) with atomic temp-file writes."""
    ),
    code(
        r"""%%capture
!pip install -q --upgrade huggingface_hub tqdm scikit-learn pillow"""
    ),
    code(
        r"""from __future__ import annotations

import hashlib
import os
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from huggingface_hub import login, snapshot_download
from PIL import Image
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "GB",
    )
else:
    DEVICE = torch.device("cpu")
    raise SystemExit(
        "GPU required for this notebook — Colab: Runtime -> Change runtime type -> GPU."
    )"""
    ),
    code(
        r"""try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ImportError:
    print("Outside Colab — ensure DRIVE_ROOT/CKPT_DIR exist.")"""
    ),
    code(
        r"""import os

from huggingface_hub import login

TOKEN = os.environ.get("HF_TOKEN", "").strip()
if TOKEN:
    login(token=TOKEN)
else:
    login()
print("Hugging Face login OK.")"""
    ),
    code(
        r"""HF_DATASET_REPO = "WatermelonAnh/FoodClassifierL1"

DRIVE_ROOT = "/content/drive/MyDrive/FitUpNutritionL1"
CKPT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
DATA_DIR = "/content/l1_dataset"

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

NUM_CLASSES = len(L1_CLASSES)
IMG_SIZE = 224
THRESHOLD = 0.5

BATCH_SIZE = 32
EPOCHS_S1, EPOCHS_S2 = 30, 20
LR_S1, LR_S2 = 1e-3, 1e-4
WEIGHT_DECAY = 1e-4
HEAD_DROPOUT = 0.4
PATIENCE = 10

NUM_WORKERS = 2
USE_AMP = True

LAST_CKPT_PATH = os.path.join(CKPT_DIR, "latest.pt")
STAGE1_BEST_PATH = os.path.join(CKPT_DIR, "stage1_best.pt")
STAGE2_BEST_PATH = os.path.join(CKPT_DIR, "best.pt")


def _config_repr() -> str:
    parts = (
        HF_DATASET_REPO,
        ",".join(L1_CLASSES),
        str(NUM_CLASSES),
        str(IMG_SIZE),
        str(BATCH_SIZE),
        str(EPOCHS_S1),
        str(EPOCHS_S2),
        str(LR_S1),
        str(LR_S2),
        str(WEIGHT_DECAY),
        str(HEAD_DROPOUT),
        str(THRESHOLD),
        str(USE_AMP),
    )
    return "|".join(parts)


CONFIG_HASH = hashlib.sha256(_config_repr().encode()).hexdigest()[:16]
print("CONFIG_HASH =", CONFIG_HASH)

os.makedirs(CKPT_DIR, exist_ok=True)"""
    ),
    code(
        r"""import pathlib

FORCE_REDOWNLOAD = False

_train = pathlib.Path(DATA_DIR) / "train"
_val = pathlib.Path(DATA_DIR) / "val"
_test = pathlib.Path(DATA_DIR) / "test"

need_dl = (
    FORCE_REDOWNLOAD
    or not _train.is_dir()
    or not _val.is_dir()
    or not _test.is_dir()
    or not any(_train.iterdir())
    or not any(_val.iterdir())
    or not any(_test.iterdir())
)

if need_dl:
    print("Downloading from HF:", HF_DATASET_REPO, "to", DATA_DIR)
    snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset", local_dir=DATA_DIR)
else:
    print("Using cached DATA_DIR:", DATA_DIR)

print("# train dirs:", sorted(p.name for p in _train.iterdir() if p.is_dir()))
print("# val dirs: ", sorted(p.name for p in _val.iterdir() if p.is_dir()))
print("# test dirs:", sorted(p.name for p in _test.iterdir() if p.is_dir()))"""
    ),
    code(
        r"""ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class FoodSceneDataset(Dataset):
    # Image folder per class: single-positive one-hot labels.

    def __init__(self, root: str | Path, tfm=None):
        self.samples: list[tuple[Path, torch.Tensor]] = []
        self.tfm = tfm
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(root_path)

        for cls_idx, cls_name in enumerate(L1_CLASSES):
            cls_dir = root_path / cls_name
            if not cls_dir.exists():
                continue
            imgs = sorted(
                p for p in cls_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXT
            )
            for p in imgs:
                y = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                y[cls_idx] = 1.0
                self.samples.append((p, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, label


transform_train = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

transform_eval = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

train_ds = FoodSceneDataset(Path(DATA_DIR) / "train", transform_train)
val_ds = FoodSceneDataset(Path(DATA_DIR) / "val", transform_eval)


def build_loader(ds, shuffle: bool) -> DataLoader:
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


train_dl = build_loader(train_ds, True)
val_dl = build_loader(val_ds, False)
print(len(train_dl), "train batches,", len(val_dl), "val batches")"""
    ),
    code(
        r"""def build_model(freeze_backbone: bool) -> nn.Module:
    wts = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    net = models.efficientnet_b4(weights=wts)

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=HEAD_DROPOUT),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return net.to(DEVICE)


def save_checkpoint_blob(state: dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint_blob(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cpu")
    old = blob.get("config_hash")
    if old is not None and old != CONFIG_HASH:
        raise RuntimeError(
            "CONFIG_HASH mismatch: checkpoint=%s current=%s. Remove old checkpoints or "
            "align config." % (old, CONFIG_HASH)
        )
    return blob


def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    preds_ls: list[np.ndarray] = []
    labels_ls: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            bin_preds = (probs >= THRESHOLD).detach().cpu().numpy()
            preds_ls.append(bin_preds)
            labels_ls.append(labels.numpy())

    y_pred = np.vstack(preds_ls)
    y_true = np.vstack(labels_ls)
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))"""
    ),
    code(
        r"""CRITERION = nn.BCEWithLogitsLoss()


def train_one_epoch(
    model,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> float:
    model.train()
    losses: list[float] = []
    use_amp = USE_AMP and DEVICE.type == "cuda"

    pbar = tqdm(dl, leave=False)
    for imgs, tgt in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                logits = model(imgs)
                loss = CRITERION(logits, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = CRITERION(logits, tgt)
            loss.backward()
            optimizer.step()

        lf = loss.detach().float().item()
        losses.append(lf)
        pbar.set_postfix(loss=lf)

    return float(sum(losses) / max(1, len(losses))))


def run_stage(
    *,
    stage: str,
    freeze_backbone: bool,
    epochs: int,
    lr: float,
    best_path: str,
    init_weights_path: str | None = None,
):
    # Curriculum stage trainer. Resume if LAST_CKPT_PATH matches stage; clear latest on OK exit.

    assert stage in {"s1", "s2"}

    model = build_model(freeze_backbone=freeze_backbone)

    resume = load_checkpoint_blob(LAST_CKPT_PATH)
    resumed = resume is not None and resume.get("stage") == stage

    scaler = GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    if resumed:
        assert resume is not None
        load_res = model.load_state_dict(resume["model"], strict=True)
        if load_res.missing_keys or load_res.unexpected_keys:
            print("[warn] state_dict mismatch:", load_res)

        best_val = float(resume["best_val"])
        patience_ctr = int(resume["patience_counter"])
        start_epoch = int(resume["epoch"]) + 1

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters())
            if freeze_backbone
            else model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        opt.load_state_dict(resume["optimizer"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs)
        )
        if "scheduler" in resume and resume["scheduler"] is not None:
            scheduler.load_state_dict(resume["scheduler"])

        if "scaler" in resume and resume["scaler"] is not None:
            scaler.load_state_dict(resume["scaler"])
    else:
        if stage == "s2":
            if not init_weights_path or not os.path.isfile(init_weights_path):
                raise FileNotFoundError(
                    "Stage 2 cold start requires STAGE1_BEST checkpoint at "
                    + str(init_weights_path)
                )

        if stage == "s2":
            ck = torch.load(init_weights_path, map_location="cpu")
            load_res = model.load_state_dict(ck["model"], strict=True)
            if load_res.missing_keys or load_res.unexpected_keys:
                print("[warn] loading stage1_best:", load_res)

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters())
            if freeze_backbone
            else model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        best_val = 0.0
        patience_ctr = 0
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        tl = train_one_epoch(model, train_dl, opt, scaler)
        scheduler.step()
        vf = evaluate(model, val_dl)
        print(
            "Stage %s | epoch %03d/%d | train_loss=%.5f | val_micro_f1=%.5f"
            % (stage, epoch + 1, epochs, tl, vf)
        )

        improved = vf > best_val
        if improved:
            best_val = vf
            patience_ctr = 0
            save_checkpoint_blob({"model": model.state_dict()}, best_path + ".candidate")
            os.replace(best_path + ".candidate", best_path)
        else:
            patience_ctr += 1

        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "patience_counter": patience_ctr,
            "stage": stage,
            "config_hash": CONFIG_HASH,
            "epochs_planned": epochs,
            "lr_planned": lr,
        }
        save_checkpoint_blob(ckpt, LAST_CKPT_PATH)

        if patience_ctr >= PATIENCE:
            print("Early stop at epoch %d | patience=%d" % (epoch + 1, PATIENCE))
            break

    if os.path.isfile(LAST_CKPT_PATH):
        try:
            os.remove(LAST_CKPT_PATH)
        except OSError as e:
            print("could not rm latest.ckpt:", e)

    print("Stage %s done. Best val_micro_f1=%.5f | best weights: %s" % (stage, best_val, best_path))
    return model, best_val


print("Trainer helpers ready.")"""
    ),
    code(
        r"""print("\\n===== Stage 1: frozen backbone, train head =====")
run_stage(
    stage="s1",
    freeze_backbone=True,
    epochs=EPOCHS_S1,
    lr=LR_S1,
    best_path=STAGE1_BEST_PATH,
)"""
    ),
    code(
        r"""print("\\n===== Stage 2: unfreeze EfficientNet =====")
torch.cuda.empty_cache()
run_stage(
    stage="s2",
    freeze_backbone=False,
    epochs=EPOCHS_S2,
    lr=LR_S2,
    best_path=STAGE2_BEST_PATH,
    init_weights_path=STAGE1_BEST_PATH,
)

if not os.path.isfile(STAGE2_BEST_PATH):
    raise RuntimeError("Stage 2 finished but best.pt missing; check Drive permissions.")"""
    ),
    code(
        r"""test_ds = FoodSceneDataset(Path(DATA_DIR) / "test", transform_eval)
test_dl = build_loader(test_ds, shuffle=False)


def load_best_model_for_eval() -> nn.Module:
    m = build_model(freeze_backbone=False)
    if not os.path.isfile(STAGE2_BEST_PATH):
        raise FileNotFoundError(STAGE2_BEST_PATH)

    ck = torch.load(STAGE2_BEST_PATH, map_location="cpu")
    m.load_state_dict(ck["model"])
    m.eval()
    return m


eval_model = load_best_model_for_eval()

preds_ls: list[np.ndarray] = []
labels_ls: list[np.ndarray] = []

with torch.no_grad():
    for imgs, labs in tqdm(test_dl, desc="test"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = eval_model(imgs)
        probs = torch.sigmoid(logits)
        b = (probs >= THRESHOLD).cpu().numpy()
        preds_ls.append(b)
        labels_ls.append(labs.numpy())

y_pred = np.vstack(preds_ls)
y_true = np.vstack(labels_ls)

print(classification_report(y_true, y_pred, target_names=L1_CLASSES, zero_division=0))

mcm = multilabel_confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.ravel()):
    sns.heatmap(
        mcm[i],
        annot=True,
        fmt="d",
        ax=ax,
        cbar=False,
        xticklabels=("Neg", "Pos"),
        yticklabels=("Neg", "Pos"),
    )
    ax.set_title(L1_CLASSES[i])
plt.tight_layout()

fig_path = os.path.join(DRIVE_ROOT, "multilabel_confusion_matrix.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig.savefig(fig_path, dpi=110)
plt.show()
print("Saved:", fig_path)"""
    ),
    code(
        r"""MODEL_FOR_INFERENCE = eval_model.to(DEVICE)


def predict(
    image_path: str | Path,
    model: nn.Module | None = None,
) -> tuple[list[float], dict[str, float]]:
    # Tuple: (scores in L1_CLASSES order, all-class probabilities dict).

    md = MODEL_FOR_INFERENCE if model is None else model
    md.eval()
    pil = Image.open(image_path).convert("RGB")
    x = transform_eval(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = md(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    d = {c: round(float(probs[i]), 4) for i, c in enumerate(L1_CLASSES)}
    return [float(v) for v in probs], d


# Smoke test — pick first file from test set if any
_demo = getattr(test_ds, "samples", [])
if _demo:
    arr, probs = predict(Path(_demo[0][0]))
    print("Example image:", Path(_demo[0][0]).name)
    print("probability array:", arr)
    print("dict:", probs)"""
    ),
    markdown(
        r"""### (Optional) Push `best.pt` to a Hugging Face model repo

Uncomment/adapt:

```python
from huggingface_hub import HfApi

HF_MODEL_REPO_ID = "WatermelonAnh/your-foodclassifier-l1-weights"

api = HfApi()

api.upload_file(
    repo_id=HF_MODEL_REPO_ID,
    path_or_fileobj=STAGE2_BEST_PATH,
    repo_type="model",
    path_in_repo="best.pt",
)
```"""
    ),
]


def main():
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
        "cells": CELLS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
