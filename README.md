# FitUpNutrition L1 — Food scene classifier training

Train an **EfficientNet-B4** multi-label classifier (8 L1 categories) following the methodology in `TRAINING_GUIDE.md`. The Colab notebook loads data from Hugging Face and saves checkpoints to **Google Drive** so runs survive disconnects.

## Dataset on Hugging Face

Upload `train/`, `val/`, and `test/` to the datasets repo **`WatermelonAnh/FoodClassifierL1`** (see [HF dataset page](https://huggingface.co/datasets/WatermelonAnh/FoodClassifierL1)) with folder-per-class layout:

```
FoodClassifierL1/
  train/
    noodle_dish/
    rice_dish/
    ...
  val/
    ...
  test/
    ...
```

## Colab notebook

Open or upload [`notebooks/train_l1_efficientnetb4.ipynb`](notebooks/train_l1_efficientnetb4.ipynb):

1. **Runtime → GPU** (T4 or better recommended).
2. Run **drive mount + Hugging Face login** cells (`HF_TOKEN` with read access for the dataset repo).
3. Adjust `HF_DATASET_REPO`, `DRIVE_ROOT`, hyperparameters if needed (defaults match the guide: 224 resize, AdamW two-stage schedule).
4. Checkpoints land under `{DRIVE_ROOT}/checkpoints/`:
   - `latest.pt` — updated every epoch; used to resume (`s1` / `s2` stage-aware).
   - `stage1_best.pt` — best Stage 1 by validation micro-F1.
   - `best.pt` — best Stage 2 (final deployable weights).

After a Stage finishes successfully, `latest.pt` is removed so a full re-run moves to the next Stage without falsely resuming the previous stage.

Notebook source-of-truth is [`scripts/gen_train_notebook.py`](scripts/gen_train_notebook.py). Regenerate the `.ipynb` after editing:

```bash
python3 scripts/gen_train_notebook.py
```

## Local dev (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Then port notebook logic to a script if needed — primary path is Colab.
```
