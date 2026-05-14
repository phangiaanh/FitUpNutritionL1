# Switch FitUpNutritionL1 from EfficientNet-B4 classifier to YOLO11s detector

Date: 2026-05-14

## Goal

Replace the current EfficientNet-B4 multi-label classifier pipeline with a YOLO11s object-detection pipeline. Same 8 L1 food categories, same Hugging Face dataset repo, new dataset payload (two tar archives in YOLO format). Final artifact is a quantized `.tflite` file for on-device mobile inference.

## Constraints and priorities

- **Framework:** Ultralytics YOLO (`ultralytics>=8.3.0`), model `yolo11s.pt` pretrained on COCO.
- **Compute:** Training on Google Colab with an A100 GPU. Notebook remains the primary execution surface (consistent with the current Colab-first workflow).
- **Deployment target:** TFLite for mobile. Accuracy is prioritized over speed during training, but model size and final inference latency on-device must remain mobile-friendly. Start with YOLO11s (INT8 ≈ 6 MB); escalate to YOLO11m only if measured mAP is insufficient.
- **Persistence:** Checkpoints, plots, and exports land on Google Drive so a Colab disconnect never destroys progress.

## Paths and config variables

Defined once in an early notebook cell, referenced throughout this spec:

```python
DRIVE_ROOT = "/content/drive/MyDrive/FitUpNutritionL1"
RUNS_DIR   = os.path.join(DRIVE_ROOT, "runs")            # Ultralytics writes weights/plots here
EXPORTS_DIR= os.path.join(DRIVE_ROOT, "exports")          # final .tflite files copied here

HF_DATASET_REPO = "WatermelonAnh/FoodClassifierL1"
HF_CACHE   = "/content/hf_cache"                          # snapshot_download target
DATA_DIR   = "/content/l1_dataset"                        # tars extracted to here
DATA_YAML  = os.path.join(DATA_DIR, "data.yaml")

RUN_NAME   = "l1_yolo11s"
```

Two-stage path separation: tar archives land in `HF_CACHE`, then are extracted into `DATA_DIR` so that `DATA_DIR` contains exactly the `images/` and `labels/` trees Ultralytics expects (and nothing else).

## Dataset

### Source

Hugging Face dataset repo `WatermelonAnh/FoodClassifierL1` (unchanged repo id). The repo now contains two tar archives instead of a folder-per-class tree:

- `images.tar` → extracts to `images/{train,val,test}/<basename>.<ext>`
- `labels.tar` → extracts to `labels/{train,val,test}/<basename>.txt`

Each `.txt` follows the YOLO standard: one line per box, `class_id cx cy w h` where coordinates are normalized to `[0.0, 1.0]` and `class_id` is an integer in `[0, 7]`.

### Class list (fixed order, index 0..7)

```
noodle_dish, rice_dish, soup_stew, grilled_fried,
banh_bread, beverage, fruit, dessert_snack
```

Hardcoded in the notebook to match the existing `L1_CLASSES` ordering. No reliance on a `classes.txt` inside the dataset.

### Extraction flow (runs in-notebook on first launch)

1. `snapshot_download(repo_id="WatermelonAnh/FoodClassifierL1", repo_type="dataset", local_dir=HF_CACHE)`.
2. Locate `images.tar` and `labels.tar` inside `HF_CACHE`; extract each into `DATA_DIR` so the tree becomes:
   ```
   DATA_DIR/
     images/{train,val,test}/*.{jpg,jpeg,png,...}
     labels/{train,val,test}/*.txt
   ```
3. Skip re-download and re-extraction when those folders are already populated. Provide a `FORCE_REDOWNLOAD = False` flag mirroring the current notebook's idiom.
4. Generate `DATA_DIR/data.yaml`:
   ```yaml
   path: <DATA_DIR>
   train: images/train
   val:   images/val
   test:  images/test
   nc: 8
   names: [noodle_dish, rice_dish, soup_stew, grilled_fried,
           banh_bread, beverage, fruit, dessert_snack]
   ```
   Ultralytics resolves labels by swapping `images/` → `labels/` in the configured paths, so no further wiring is needed.

### Label validation (mandatory pre-flight)

Before training starts, walk every `.txt` under `labels/{train,val,test}` and assert:

- The file is allowed to be empty (background image with no objects). Non-empty files must contain one or more whitespace-delimited lines with **exactly 5 fields** each.
- Field 0 parses as `int` and is in `[0, 7]`.
- Fields 1–4 parse as `float` and are in `[0.0, 1.0]`.
- Every image in `images/<split>/` has a sibling `.txt` in `labels/<split>/` with the matching basename (and vice versa). Report mismatches.

Print a summary table per split: number of label files, total boxes, boxes-per-class histogram. **Fail fast** on the first violation so dataset issues are caught before a long training run.

## Model and training

### Configuration

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # COCO-pretrained
model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=100,
    batch=64,
    patience=30,
    workers=8,
    cache="ram",        # ~80 GB Colab A100 RAM is usually enough; fall back to "disk" if OOM
    amp=True,
    device=0,
    project=RUNS_DIR,   # Drive path so weights survive disconnects
    name="l1_yolo11s",
    exist_ok=True,
)
```

- **Single-stage fine-tune.** The classifier's freeze→unfreeze two-stage schedule is dropped; YOLO fine-tuning works well as one stage with the framework's built-in warmup, cosine LR, and mosaic augmentation.
- **AutoBatch is not used.** Batch is fixed at 64 for reproducibility on A100.
- **Optimizer** is left at Ultralytics' `auto` setting (the framework picks AdamW for small datasets, SGD for large ones).
- **Augmentation** uses Ultralytics defaults (mosaic, mixup off, hsv jitter, fliplr).

### Resume after disconnect

A separate cell checks for `RUNS_DIR/l1_yolo11s/weights/last.pt`; if present, re-instantiate the run with `model.train(..., resume=True)`. The `last.pt` file is written by Ultralytics every epoch and lives on Drive, so it survives runtime resets.

## Evaluation

```python
metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=64,
)
```

Reported and persisted:

- Overall **mAP50**, **mAP50-95**, precision, recall.
- Per-class AP (8 rows).
- Auto-saved plots under `RUNS_DIR/l1_yolo11s/`: `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `results.csv`.

No micro-F1, no multilabel confusion matrix — those were classifier-specific.

## TFLite export (mobile deployment)

Two export cells run after training, both consuming `RUNS_DIR/l1_yolo11s/weights/best.pt`:

### Primary: INT8 quantized

```python
model = YOLO(f"{RUNS_DIR}/l1_yolo11s/weights/best.pt")
model.export(
    format="tflite",
    int8=True,
    imgsz=640,
    data=DATA_YAML,  # required for INT8 calibration sampling
)
```

Produces `best_int8.tflite` (~6 MB). The `data` argument lets Ultralytics draw calibration images from the configured training set to fit INT8 activation ranges.

### Fallback: FP16

```python
model.export(format="tflite", half=True, imgsz=640)
```

Produces `best_float16.tflite` (~22 MB). Use this if INT8 calibration causes accuracy regressions.

Both `.tflite` files are copied to `DRIVE_ROOT/exports/` for easy download.

## Inference smoke test

Final notebook cell uses the **TFLite file** (not the `.pt`) so we verify the actual deployment artifact:

1. Pick a random image from `images/test/`.
2. Run `YOLO("best_int8.tflite").predict(img_path, imgsz=640, conf=0.25)`.
3. Annotate with class names and confidence; save preview PNG to `DRIVE_ROOT/exports/smoke_test.png`.

## Repo changes

### Replaced

- `scripts/gen_train_notebook.py` — rewritten end-to-end. The `CELLS` list now produces a detection notebook instead of the classifier notebook. File path unchanged so the README's "regenerate" instruction stays valid.
- `notebooks/train_l1_efficientnetb4.ipynb` — deleted.
- `notebooks/train_l1_yolo11s.ipynb` — new, generated by `scripts/gen_train_notebook.py`.
- `README.md` — updated dataset description (tar layout, YOLO labels), notebook name, metrics (mAP instead of micro-F1), and reflects the TFLite export step.
- `requirements.txt` — add `ultralytics>=8.3.0`; remove `scikit-learn` (replaced by Ultralytics' metrics). Keep `huggingface_hub`, `tqdm`, `pillow`. `torch` / `torchvision` pinned versions stay (Ultralytics depends on them).

### Removed (no longer needed)

- `FoodSceneDataset`, the train/eval transforms, `BCEWithLogitsLoss`, `run_stage`, two-stage curriculum, `CONFIG_HASH` blob, micro-F1 / multilabel confusion matrix utilities — all replaced by Ultralytics' built-in equivalents.

## Notebook cell outline (final)

The regenerated `train_l1_yolo11s.ipynb` will be organized as:

1. **Markdown header** — purpose, dataset, classes, deployment target.
2. **`pip install`** — `ultralytics`, `huggingface_hub`.
3. **Imports + GPU check** — fail loudly if not on a GPU runtime.
4. **Drive mount + HF login** — same pattern as the current notebook.
5. **Config** — paths, class list, hyperparameters.
6. **Dataset download + extract + data.yaml generation.**
7. **Label validation pre-flight.**
8. **Train** (with resume detection).
9. **Evaluate on test split.**
10. **Export TFLite (INT8 + FP16).**
11. **Inference smoke test on TFLite.**

## Out of scope

- Publishing weights to a Hugging Face model repo (the current notebook has a commented-out cell for this; the new notebook will not include it).
- Hyperparameter search / sweeps.
- On-device benchmarking of the exported TFLite (separate concern, lives in the mobile app project).
- Switching to YOLO11m / YOLO11l — only revisit if measured 11s mAP on test set falls below the project's accuracy target.

## Success criteria

1. The regenerated notebook runs end-to-end on a fresh Colab A100 with `HF_TOKEN` set, producing `best.pt`, `best_int8.tflite`, and `best_float16.tflite` on Drive.
2. Test-set mAP50 is reported and printed; per-class AP is visible for all 8 classes.
3. The label-validation cell rejects malformed `.txt` files with a clear error before training starts.
4. After an intentional runtime reset, re-running the notebook resumes from `last.pt` rather than restarting from epoch 0.
5. The inference smoke test produces a correctly annotated preview image using the INT8 TFLite file.
