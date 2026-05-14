# FitUpNutrition L1 - Food scene detector training

Train a **YOLO11s** object detector on 8 L1 food categories. The Colab notebook
loads data from Hugging Face and saves checkpoints + TFLite exports to
**Google Drive** so runs survive disconnects.

## Dataset on Hugging Face

Repo **`WatermelonAnh/FoodClassifierL1`** ([HF dataset page](https://huggingface.co/datasets/WatermelonAnh/FoodClassifierL1))
holds two tar archives:

```
FoodClassifierL1/
  images.tar     -> images/{train,val,test}/<basename>.<ext>
  labels.tar     -> labels/{train,val,test}/<basename>.txt
```

Labels follow the **YOLO standard**: one line per box,
`class_id cx cy w h` with coordinates normalized to `[0.0, 1.0]` and
`class_id` an integer in `[0, 7]`.

Classes (fixed order, index 0..7):
`noodle_dish`, `rice_dish`, `soup_stew`, `grilled_fried`, `banh_bread`,
`beverage`, `fruit`, `dessert_snack`.

## Colab notebook

Open or upload [`notebooks/train_l1_yolo11s.ipynb`](notebooks/train_l1_yolo11s.ipynb):

1. **Runtime → GPU** (A100 recommended; the defaults assume A100 80GB).
2. Run **drive mount + Hugging Face login** cells (`HF_TOKEN` with read access for the dataset repo).
3. Adjust hyperparameters if needed (defaults: `imgsz=640`, `epochs=100`,
   `batch=64`, `patience=30`, `cache="ram"`).
4. Artifacts land under `/content/drive/MyDrive/FitUpNutritionL1/`:
   - `runs/l1_yolo11s/weights/best.pt` — best validation checkpoint.
   - `runs/l1_yolo11s/weights/last.pt` — most recent epoch (used to resume).
   - `runs/l1_yolo11s/results.csv`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png` — training plots.
   - `exports/best_int8.tflite` — INT8 quantized TFLite (~6 MB, primary mobile artifact).
   - `exports/best_float16.tflite` — FP16 TFLite (~22 MB, fallback).
   - `exports/smoke_test.png` — annotated inference preview.

If a Colab session disconnects mid-run, just re-run the notebook: the train
cell auto-detects `last.pt` on Drive and resumes.

Test-set evaluation reports mAP50, mAP50-95, precision, recall, and per-class
AP. No micro-F1 (that was the old classifier).

## Regenerating the notebook

The notebook source-of-truth is [`scripts/gen_train_notebook.py`](scripts/gen_train_notebook.py).
Helper functions (`validate_yolo_labels`, `extract_dataset_tars`,
`write_data_yaml`) live in [`scripts/notebook_helpers.py`](scripts/notebook_helpers.py)
and are embedded verbatim into one notebook cell so the notebook stays
self-contained in Colab.

Regenerate the `.ipynb` after editing either file:

```bash
python3 scripts/gen_train_notebook.py
```

## Tests

```bash
python -m unittest discover tests -v
```

Tests cover the helper functions (label validation, tar extraction, yaml
emission) and the notebook generator (every code cell parses as Python,
expected sections are present).

## Local dev (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Training itself runs on Colab; locally you can run the generator + tests.
```
