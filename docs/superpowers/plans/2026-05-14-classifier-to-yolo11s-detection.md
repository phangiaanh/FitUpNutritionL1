# YOLO11s Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the EfficientNet-B4 multi-label classifier notebook with a YOLO11s object-detection notebook that consumes the new HF tar-archive dataset (YOLO label format), trains on a Colab A100, and exports INT8 + FP16 TFLite artifacts for mobile inference.

**Architecture:** A notebook generator (`scripts/gen_train_notebook.py`) emits `notebooks/train_l1_yolo11s.ipynb`. Pure helper functions for dataset extraction, label validation, and `data.yaml` generation live in `scripts/notebook_helpers.py` so they can be unit-tested locally; the generator embeds that file's source verbatim into a single notebook cell so the notebook remains self-contained when run in Colab.

**Tech Stack:** Python 3.10+, Ultralytics YOLO (`ultralytics>=8.3.0`), `huggingface_hub`, `pillow`, `tqdm`. Tests use stdlib `unittest` (no new dev dependency). Notebook runs on Google Colab with an A100 GPU.

---

## Reference: spec

This plan implements `docs/superpowers/specs/2026-05-14-classifier-to-yolo11s-detection-design.md`.

The full spec is the source of truth for behavior. Each task below cites which spec section it covers.

---

## Task 1: Update requirements.txt and create test scaffold

Spec sections: *Repo changes → Replaced* (requirements.txt).

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Read current requirements.txt**

Run: `cat requirements.txt`
Expected output:
```
# Local / Colab-aligned pins (Colab ships torch + torchvision separately).
torch>=2.1.0
torchvision>=0.16.0
huggingface_hub>=0.21.0
tqdm>=4.66.0
scikit-learn>=1.3.0
pillow>=10.0.0
```

- [ ] **Step 2: Rewrite `requirements.txt`**

Replace the entire file with:

```
# Local / Colab-aligned pins (Colab ships torch + torchvision separately).
torch>=2.1.0
torchvision>=0.16.0
ultralytics>=8.3.0
huggingface_hub>=0.21.0
tqdm>=4.66.0
pillow>=10.0.0
pyyaml>=6.0
```

Rationale: `ultralytics` is the new training framework. `scikit-learn` is removed (its metrics are no longer used — Ultralytics computes mAP internally). `pyyaml` is added because the helpers will write `data.yaml`. `torch` / `torchvision` stay because Ultralytics depends on them.

- [ ] **Step 3: Create empty test package marker**

Create `tests/__init__.py` with empty content (zero bytes is fine; Python treats it as a package marker).

- [ ] **Step 4: Verify the file changes**

Run: `cat requirements.txt && ls tests/`
Expected: new requirements content prints; `tests/` directory exists containing `__init__.py`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "deps: swap scikit-learn for ultralytics, add pyyaml; scaffold tests/"
```

---

## Task 2: Implement and test `validate_yolo_labels`

Spec section: *Dataset → Label validation (mandatory pre-flight)*.

**Files:**
- Create: `scripts/notebook_helpers.py`
- Create: `tests/test_notebook_helpers.py`

- [ ] **Step 1: Write failing tests in `tests/test_notebook_helpers.py`**

Create `tests/test_notebook_helpers.py`:

```python
"""Unit tests for scripts/notebook_helpers.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.notebook_helpers import validate_yolo_labels  # noqa: E402


CLASSES = [
    "noodle_dish",
    "rice_dish",
    "soup_stew",
    "grilled_fried",
    "banh_bread",
    "beverage",
    "fruit",
    "dessert_snack",
]


def _make_split(root: Path, split: str, pairs: list[tuple[str, str]]) -> None:
    """pairs is a list of (image_basename, label_text). Image is touched (empty file)."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for base, lbl_text in pairs:
        (img_dir / f"{base}.jpg").write_bytes(b"")
        (lbl_dir / f"{base}.txt").write_text(lbl_text)


class ValidateYoloLabelsTests(unittest.TestCase):
    def test_accepts_well_formed_dataset(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [
                ("a", "0 0.5 0.5 0.4 0.4\n"),
                ("b", "7 0.1 0.1 0.2 0.2\n2 0.9 0.9 0.05 0.05\n"),
            ])
            _make_split(root, "val", [("c", "3 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "")])  # empty label = background, allowed
            validate_yolo_labels(root, CLASSES)  # should not raise

    def test_rejects_class_id_out_of_range(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [("a", "8 0.5 0.5 0.4 0.4\n")])
            _make_split(root, "val", [("c", "0 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "class id"):
                validate_yolo_labels(root, CLASSES)

    def test_rejects_coord_out_of_unit_interval(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [("a", "0 1.5 0.5 0.4 0.4\n")])
            _make_split(root, "val", [("c", "0 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "coord"):
                validate_yolo_labels(root, CLASSES)

    def test_rejects_wrong_field_count(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [("a", "0 0.5 0.5 0.4\n")])  # only 4 fields
            _make_split(root, "val", [("c", "0 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "5 fields"):
                validate_yolo_labels(root, CLASSES)

    def test_rejects_non_integer_class_id(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [("a", "0.5 0.5 0.5 0.4 0.4\n")])
            _make_split(root, "val", [("c", "0 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "class id"):
                validate_yolo_labels(root, CLASSES)

    def test_reports_image_without_label(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [("a", "0 0.5 0.5 0.5 0.5\n")])
            # Add an orphan image without label
            (root / "images" / "train" / "orphan.jpg").write_bytes(b"")
            _make_split(root, "val", [("c", "0 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "orphan"):
                validate_yolo_labels(root, CLASSES)

    def test_reports_label_without_image(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_split(root, "train", [("a", "0 0.5 0.5 0.5 0.5\n")])
            (root / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.5 0.5\n")
            _make_split(root, "val", [("c", "0 0.5 0.5 0.5 0.5\n")])
            _make_split(root, "test", [("d", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "ghost"):
                validate_yolo_labels(root, CLASSES)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Create stub `scripts/notebook_helpers.py` and verify tests fail**

Create `scripts/notebook_helpers.py`:

```python
"""Pure helpers for the YOLO11s detection notebook.

These functions are kept in a standalone module so they can be unit-tested
locally. The notebook generator embeds this file's source verbatim into one
notebook cell so the notebook stays self-contained when run in Colab.
"""

from __future__ import annotations


def validate_yolo_labels(data_dir, class_names) -> None:
    raise NotImplementedError
```

Run: `python -m unittest tests.test_notebook_helpers -v`
Expected: every test fails with `NotImplementedError`.

- [ ] **Step 3: Implement `validate_yolo_labels`**

Replace `scripts/notebook_helpers.py` with:

```python
"""Pure helpers for the YOLO11s detection notebook.

These functions are kept in a standalone module so they can be unit-tested
locally. The notebook generator embeds this file's source verbatim into one
notebook cell so the notebook stays self-contained when run in Colab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def validate_yolo_labels(
    data_dir,
    class_names: list[str],
    splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    """Walk every YOLO label file under data_dir/labels/<split>/ and assert
    well-formedness. Raises ValueError on the first violation.

    On success, prints a per-split summary: file count, total boxes, and a
    boxes-per-class histogram.

    A `.txt` file may be empty (image with no objects). Non-empty files must
    have lines of exactly 5 whitespace-separated fields:
        class_id  cx  cy  w  h
    where class_id is an integer in [0, len(class_names)-1] and cx/cy/w/h are
    floats in [0.0, 1.0].

    Every image in images/<split>/ must have a matching basename .txt in
    labels/<split>/ and vice versa.
    """
    data_dir = Path(data_dir)
    nc = len(class_names)

    for split in splits:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        if not img_dir.is_dir():
            raise ValueError(f"missing image dir: {img_dir}")
        if not lbl_dir.is_dir():
            raise ValueError(f"missing label dir: {lbl_dir}")

        img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"}

        orphans = sorted(img_stems - lbl_stems)
        ghosts = sorted(lbl_stems - img_stems)
        if orphans:
            raise ValueError(f"[{split}] images without labels (orphan): {orphans[:5]}")
        if ghosts:
            raise ValueError(f"[{split}] labels without images (ghost): {ghosts[:5]}")

        per_class = [0] * nc
        total_boxes = 0
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            text = lbl_path.read_text().strip()
            if not text:
                continue
            for line_idx, raw in enumerate(text.splitlines(), start=1):
                fields = raw.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"[{split}] {lbl_path.name}:{line_idx}: expected 5 fields, "
                        f"got {len(fields)}"
                    )
                try:
                    cls = int(fields[0])
                    if str(cls) != fields[0]:
                        raise ValueError("not integer")
                except ValueError:
                    raise ValueError(
                        f"[{split}] {lbl_path.name}:{line_idx}: class id not an integer "
                        f"({fields[0]!r})"
                    )
                if cls < 0 or cls >= nc:
                    raise ValueError(
                        f"[{split}] {lbl_path.name}:{line_idx}: class id {cls} out of "
                        f"range [0, {nc - 1}]"
                    )
                try:
                    coords = [float(x) for x in fields[1:]]
                except ValueError:
                    raise ValueError(
                        f"[{split}] {lbl_path.name}:{line_idx}: non-numeric coord in "
                        f"{fields[1:]}"
                    )
                for c in coords:
                    if c < 0.0 or c > 1.0:
                        raise ValueError(
                            f"[{split}] {lbl_path.name}:{line_idx}: coord {c} out of "
                            f"[0.0, 1.0]"
                        )
                per_class[cls] += 1
                total_boxes += 1

        print(f"[{split}] files={len(lbl_stems)} boxes={total_boxes}")
        for i, name in enumerate(class_names):
            print(f"  {i} {name}: {per_class[i]}")
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `python -m unittest tests.test_notebook_helpers -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/notebook_helpers.py tests/test_notebook_helpers.py
git commit -m "feat(helpers): add validate_yolo_labels with unit tests"
```

---

## Task 3: Implement and test `extract_dataset_tars`

Spec section: *Dataset → Extraction flow* (steps 1–3).

**Files:**
- Modify: `scripts/notebook_helpers.py`
- Modify: `tests/test_notebook_helpers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_notebook_helpers.py` *above* the `if __name__ == "__main__":` block:

```python
import tarfile

from scripts.notebook_helpers import extract_dataset_tars  # noqa: E402


def _make_tar(tar_path: Path, files: dict[str, bytes]) -> None:
    """Create a tar archive containing the given path→content entries."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for arcname, data in files.items():
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            import io
            tar.addfile(info, io.BytesIO(data))


class ExtractDatasetTarsTests(unittest.TestCase):
    def test_extracts_images_and_labels_tars(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_dir = Path(td) / "data"
            _make_tar(hf_cache / "images.tar", {
                "images/train/a.jpg": b"img",
                "images/val/b.jpg": b"img",
                "images/test/c.jpg": b"img",
            })
            _make_tar(hf_cache / "labels.tar", {
                "labels/train/a.txt": b"0 0.5 0.5 0.5 0.5\n",
                "labels/val/b.txt": b"0 0.5 0.5 0.5 0.5\n",
                "labels/test/c.txt": b"0 0.5 0.5 0.5 0.5\n",
            })
            extract_dataset_tars(hf_cache, data_dir)
            self.assertTrue((data_dir / "images" / "train" / "a.jpg").is_file())
            self.assertTrue((data_dir / "labels" / "val" / "b.txt").is_file())

    def test_skips_when_already_extracted(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_dir = Path(td) / "data"
            # Pre-populate the target tree so extraction should be skipped.
            for split in ("train", "val", "test"):
                (data_dir / "images" / split).mkdir(parents=True)
                (data_dir / "labels" / split).mkdir(parents=True)
                (data_dir / "images" / split / "pre.jpg").write_bytes(b"")
                (data_dir / "labels" / split / "pre.txt").write_bytes(b"")
            # Note: no tars in hf_cache; the function must not look for them.
            hf_cache.mkdir()
            extract_dataset_tars(hf_cache, data_dir)  # must not raise
            self.assertTrue((data_dir / "images" / "train" / "pre.jpg").is_file())

    def test_force_re_extracts(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_dir = Path(td) / "data"
            for split in ("train", "val", "test"):
                (data_dir / "images" / split).mkdir(parents=True)
                (data_dir / "labels" / split).mkdir(parents=True)
                (data_dir / "images" / split / "stale.jpg").write_bytes(b"old")
            _make_tar(hf_cache / "images.tar", {
                "images/train/fresh.jpg": b"new",
                "images/val/fresh.jpg": b"new",
                "images/test/fresh.jpg": b"new",
            })
            _make_tar(hf_cache / "labels.tar", {
                "labels/train/fresh.txt": b"0 0.5 0.5 0.5 0.5\n",
                "labels/val/fresh.txt": b"0 0.5 0.5 0.5 0.5\n",
                "labels/test/fresh.txt": b"0 0.5 0.5 0.5 0.5\n",
            })
            extract_dataset_tars(hf_cache, data_dir, force=True)
            self.assertTrue((data_dir / "images" / "train" / "fresh.jpg").is_file())

    def test_raises_if_tar_missing(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_dir = Path(td) / "data"
            hf_cache.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "images.tar"):
                extract_dataset_tars(hf_cache, data_dir)
```

Also extend the existing imports at the top of the file. Specifically, the line `from scripts.notebook_helpers import validate_yolo_labels  # noqa: E402` stays; the new `from scripts.notebook_helpers import extract_dataset_tars  # noqa: E402` is added inside the file (already placed in the snippet above near its tests).

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m unittest tests.test_notebook_helpers -v`
Expected: 4 new tests fail with `ImportError` (function not yet defined).

- [ ] **Step 3: Implement `extract_dataset_tars`**

Append to `scripts/notebook_helpers.py`:

```python
import tarfile


def extract_dataset_tars(hf_cache, data_dir, force: bool = False) -> None:
    """Extract images.tar and labels.tar from hf_cache into data_dir.

    Expected resulting tree:
        data_dir/
          images/{train,val,test}/...
          labels/{train,val,test}/...

    If `force` is False and the target tree already exists with at least one
    file in each of the six split dirs, extraction is skipped.

    Raises FileNotFoundError if a required tar is missing and extraction
    actually needs to happen.
    """
    hf_cache = Path(hf_cache)
    data_dir = Path(data_dir)

    splits = ("train", "val", "test")

    def already_populated() -> bool:
        for kind in ("images", "labels"):
            for split in splits:
                d = data_dir / kind / split
                if not d.is_dir():
                    return False
                if not any(d.iterdir()):
                    return False
        return True

    if not force and already_populated():
        print(f"[extract] {data_dir} already populated, skipping")
        return

    images_tar = hf_cache / "images.tar"
    labels_tar = hf_cache / "labels.tar"
    if not images_tar.is_file():
        raise FileNotFoundError(f"images.tar not found under {hf_cache}")
    if not labels_tar.is_file():
        raise FileNotFoundError(f"labels.tar not found under {hf_cache}")

    data_dir.mkdir(parents=True, exist_ok=True)
    for tar_path in (images_tar, labels_tar):
        print(f"[extract] {tar_path.name} -> {data_dir}")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(data_dir)
```

- [ ] **Step 4: Run tests, all pass**

Run: `python -m unittest tests.test_notebook_helpers -v`
Expected: all tests pass (7 existing + 4 new = 11 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/notebook_helpers.py tests/test_notebook_helpers.py
git commit -m "feat(helpers): add extract_dataset_tars with unit tests"
```

---

## Task 4: Implement and test `write_data_yaml`

Spec section: *Dataset → Extraction flow* (step 4).

**Files:**
- Modify: `scripts/notebook_helpers.py`
- Modify: `tests/test_notebook_helpers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_notebook_helpers.py` above the `if __name__ == "__main__":` block:

```python
import yaml as _yaml

from scripts.notebook_helpers import write_data_yaml  # noqa: E402


class WriteDataYamlTests(unittest.TestCase):
    def test_writes_expected_yaml(self) -> None:
        with TemporaryDirectory() as td:
            data_dir = Path(td)
            out = write_data_yaml(data_dir, CLASSES)
            self.assertEqual(out, data_dir / "data.yaml")
            doc = _yaml.safe_load(out.read_text())
            self.assertEqual(doc["path"], str(data_dir))
            self.assertEqual(doc["train"], "images/train")
            self.assertEqual(doc["val"], "images/val")
            self.assertEqual(doc["test"], "images/test")
            self.assertEqual(doc["nc"], 8)
            self.assertEqual(doc["names"], CLASSES)

    def test_overwrites_existing(self) -> None:
        with TemporaryDirectory() as td:
            data_dir = Path(td)
            (data_dir / "data.yaml").write_text("garbage: true\n")
            write_data_yaml(data_dir, CLASSES)
            doc = _yaml.safe_load((data_dir / "data.yaml").read_text())
            self.assertEqual(doc["nc"], 8)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m unittest tests.test_notebook_helpers -v`
Expected: 2 new tests fail with `ImportError`.

- [ ] **Step 3: Implement `write_data_yaml`**

Append to `scripts/notebook_helpers.py`:

```python
import yaml


def write_data_yaml(data_dir, class_names: list[str]):
    """Write Ultralytics data.yaml under data_dir. Returns the Path written.

    Overwrites any existing file at that location.
    """
    data_dir = Path(data_dir)
    out = data_dir / "data.yaml"
    doc = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": list(class_names),
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out
```

- [ ] **Step 4: Run tests, all pass**

Run: `python -m unittest tests.test_notebook_helpers -v`
Expected: all 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/notebook_helpers.py tests/test_notebook_helpers.py
git commit -m "feat(helpers): add write_data_yaml with unit tests"
```

---

## Task 5: Rewrite `scripts/gen_train_notebook.py` skeleton + intro cells

Spec sections: *Notebook cell outline* (cells 1–5), *Paths and config variables*.

**Files:**
- Modify (full rewrite): `scripts/gen_train_notebook.py`
- Create: `tests/test_notebook_generation.py`

- [ ] **Step 1: Write the failing generation test**

Create `tests/test_notebook_generation.py`:

```python
"""End-to-end test: gen_train_notebook.py emits a valid notebook."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_train_notebook.py"


class NotebookGenerationTests(unittest.TestCase):
    def test_generator_runs_and_produces_valid_json(self) -> None:
        with TemporaryDirectory() as td:
            out_path = Path(td) / "out.ipynb"
            result = subprocess.run(
                [sys.executable, str(GEN_SCRIPT), "--out", str(out_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(out_path.is_file())
            doc = json.loads(out_path.read_text())
            self.assertEqual(doc["nbformat"], 4)
            self.assertIn("cells", doc)
            self.assertGreater(len(doc["cells"]), 0)

    def test_every_code_cell_parses_as_python(self) -> None:
        with TemporaryDirectory() as td:
            out_path = Path(td) / "out.ipynb"
            subprocess.run(
                [sys.executable, str(GEN_SCRIPT), "--out", str(out_path)],
                check=True,
            )
            doc = json.loads(out_path.read_text())
            for i, cell in enumerate(doc["cells"]):
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                # Skip pip-install cells (start with %%capture or %pip / !pip).
                if src.lstrip().startswith(("%%capture", "%pip", "!pip")):
                    continue
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    self.fail(f"cell {i} failed to parse: {e}\n--- source ---\n{src}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `python -m unittest tests.test_notebook_generation -v`
Expected: tests fail (either old script doesn't accept `--out`, or imports break).

- [ ] **Step 3: Replace `scripts/gen_train_notebook.py` with the new skeleton**

Overwrite `scripts/gen_train_notebook.py`:

```python
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
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `python -m unittest tests.test_notebook_generation -v`
Expected: both `NotebookGenerationTests` pass.

- [ ] **Step 5: Generate the notebook once and inspect**

Run: `python scripts/gen_train_notebook.py`
Expected: `Wrote /Users/lap15626/source/outsource/FitUpNutritionL1/notebooks/train_l1_yolo11s.ipynb`

Then: `python -c "import json; d=json.load(open('notebooks/train_l1_yolo11s.ipynb')); print('cells:', len(d['cells']))"`
Expected: `cells: 7` (5 logical sections + 1 markdown helpers heading + 1 helpers code cell).

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_train_notebook.py notebooks/train_l1_yolo11s.ipynb tests/test_notebook_generation.py
git commit -m "feat(notebook): generator skeleton with intro cells (header, install, imports, drive/HF, config, helpers)"
```

---

## Task 6: Add dataset download + extract + data.yaml cell

Spec section: *Dataset → Extraction flow*.

**Files:**
- Modify: `scripts/gen_train_notebook.py` (extend `build_cells()`)
- Modify: `tests/test_notebook_generation.py`
- Modify: `notebooks/train_l1_yolo11s.ipynb` (regenerated)

- [ ] **Step 1: Extend the generation test**

Append a new test class inside `tests/test_notebook_generation.py` above `if __name__ == "__main__":`:

```python
class DatasetCellTests(unittest.TestCase):
    def _sources(self) -> list[str]:
        with TemporaryDirectory() as td:
            out_path = Path(td) / "out.ipynb"
            subprocess.run(
                [sys.executable, str(GEN_SCRIPT), "--out", str(out_path)],
                check=True,
            )
            doc = json.loads(out_path.read_text())
            return ["".join(c["source"]) for c in doc["cells"]]

    def test_dataset_cell_present(self) -> None:
        srcs = self._sources()
        joined = "\n".join(srcs)
        self.assertIn("snapshot_download", joined)
        self.assertIn("extract_dataset_tars", joined)
        self.assertIn("write_data_yaml", joined)
```

- [ ] **Step 2: Run, confirm it fails**

Run: `python -m unittest tests.test_notebook_generation.DatasetCellTests -v`
Expected: fails — `extract_dataset_tars`/`write_data_yaml` not yet wired up in the generator.

- [ ] **Step 3: Add the dataset cell in `build_cells()`**

In `scripts/gen_train_notebook.py`, inside `build_cells()`, append before the `return cells` statement:

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `python -m unittest discover tests -v`
Expected: all tests pass (13 helper tests + 3 generation tests = 16 total).

- [ ] **Step 5: Regenerate the notebook**

Run: `python scripts/gen_train_notebook.py`
Expected: `Wrote ...train_l1_yolo11s.ipynb`.

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_train_notebook.py tests/test_notebook_generation.py notebooks/train_l1_yolo11s.ipynb
git commit -m "feat(notebook): add dataset download/extract/data.yaml cell"
```

---

## Task 7: Add label validation cell

Spec section: *Dataset → Label validation (mandatory pre-flight)*.

**Files:**
- Modify: `scripts/gen_train_notebook.py`
- Modify: `tests/test_notebook_generation.py`
- Modify: `notebooks/train_l1_yolo11s.ipynb`

- [ ] **Step 1: Extend test**

Add to `DatasetCellTests` in `tests/test_notebook_generation.py`:

```python
    def test_label_validation_cell_present(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("validate_yolo_labels", joined)
```

- [ ] **Step 2: Run, confirm it fails**

Run: `python -m unittest tests.test_notebook_generation.DatasetCellTests.test_label_validation_cell_present -v`
Expected: fail.

- [ ] **Step 3: Add validation cell to generator**

In `build_cells()` after the dataset cell, append:

```python
    # Cell: pre-flight label validation
    cells.append(markdown(
        "### Label validation\n\n"
        "Fails loudly if any `.txt` is malformed or if there are orphan "
        "images/labels. Run this **before** training."
    ))
    cells.append(code("validate_yolo_labels(DATA_DIR, L1_CLASSES)"))
```

- [ ] **Step 4: Run all tests, regenerate notebook**

Run: `python -m unittest discover tests -v && python scripts/gen_train_notebook.py`
Expected: all tests pass; notebook regenerated.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_train_notebook.py tests/test_notebook_generation.py notebooks/train_l1_yolo11s.ipynb
git commit -m "feat(notebook): add YOLO label validation pre-flight cell"
```

---

## Task 8: Add training cell with resume detection

Spec section: *Model and training* (configuration + resume after disconnect).

**Files:**
- Modify: `scripts/gen_train_notebook.py`
- Modify: `tests/test_notebook_generation.py`
- Modify: `notebooks/train_l1_yolo11s.ipynb`

- [ ] **Step 1: Extend test**

Add a new test class to `tests/test_notebook_generation.py`:

```python
class TrainingCellTests(unittest.TestCase):
    def _sources(self) -> list[str]:
        with TemporaryDirectory() as td:
            out_path = Path(td) / "out.ipynb"
            subprocess.run(
                [sys.executable, str(GEN_SCRIPT), "--out", str(out_path)],
                check=True,
            )
            doc = json.loads(out_path.read_text())
            return ["".join(c["source"]) for c in doc["cells"]]

    def test_training_cell_uses_yolo11s_and_a100_params(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("YOLO(\"yolo11s.pt\")", joined)
        self.assertIn("model.train(", joined)
        self.assertIn("data=DATA_YAML", joined)
        self.assertIn("imgsz=IMG_SIZE", joined)
        self.assertIn("epochs=EPOCHS", joined)
        self.assertIn("batch=BATCH", joined)
        self.assertIn("patience=PATIENCE", joined)
        self.assertIn("project=RUNS_DIR", joined)
        self.assertIn("name=RUN_NAME", joined)

    def test_resume_logic_present(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("last.pt", joined)
        self.assertIn("resume=True", joined)
```

- [ ] **Step 2: Run, confirm failing**

Run: `python -m unittest tests.test_notebook_generation.TrainingCellTests -v`
Expected: both new tests fail.

- [ ] **Step 3: Add training cell to generator**

In `build_cells()`, append after the validation cell:

```python
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
resume = os.path.isfile(LAST_PT)
if resume:
    print(f"Resuming from {LAST_PT}")
    model = YOLO(LAST_PT)
else:
    print("Starting fresh from yolo11s.pt")
    model = YOLO("yolo11s.pt")

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
```

- [ ] **Step 4: Run all tests + regenerate**

Run: `python -m unittest discover tests -v && python scripts/gen_train_notebook.py`
Expected: all pass; notebook regenerated.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_train_notebook.py tests/test_notebook_generation.py notebooks/train_l1_yolo11s.ipynb
git commit -m "feat(notebook): add training cell with resume-on-last.pt logic"
```

---

## Task 9: Add test-split evaluation cell

Spec section: *Evaluation*.

**Files:**
- Modify: `scripts/gen_train_notebook.py`
- Modify: `tests/test_notebook_generation.py`
- Modify: `notebooks/train_l1_yolo11s.ipynb`

- [ ] **Step 1: Extend test**

Add to `TrainingCellTests`:

```python
    def test_eval_cell_present(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("model.val(", joined)
        self.assertIn('split="test"', joined)
        self.assertIn("mAP50", joined)
```

- [ ] **Step 2: Run, confirm failing**

Run: `python -m unittest tests.test_notebook_generation.TrainingCellTests.test_eval_cell_present -v`
Expected: fail.

- [ ] **Step 3: Add eval cell**

In `build_cells()`, append after the training cell:

```python
    # Cell: evaluate on test split
    cells.append(markdown(
        "### Evaluate on test split\n\n"
        "Reports overall mAP50, mAP50-95, precision, recall, and per-class AP. "
        "Plots (`confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, "
        "`results.csv`) auto-save under `RUNS_DIR/RUN_NAME/`."
    ))
    cells.append(code(
        r"""BEST_PT = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
if not os.path.isfile(BEST_PT):
    raise FileNotFoundError(f"best.pt not found at {BEST_PT} - did training finish?")

best_model = YOLO(BEST_PT)
metrics = best_model.val(
    data=DATA_YAML,
    split="test",
    imgsz=IMG_SIZE,
    batch=BATCH,
)

print(f"\nmAP50      = {metrics.box.map50:.4f}")
print(f"mAP50-95   = {metrics.box.map:.4f}")
print(f"precision  = {metrics.box.mp:.4f}")
print(f"recall     = {metrics.box.mr:.4f}")

print("\nPer-class AP@0.5:")
for i, name in enumerate(L1_CLASSES):
    ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
    print(f"  {i} {name}: {ap50:.4f}")"""
    ))
```

- [ ] **Step 4: Run all tests + regenerate**

Run: `python -m unittest discover tests -v && python scripts/gen_train_notebook.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_train_notebook.py tests/test_notebook_generation.py notebooks/train_l1_yolo11s.ipynb
git commit -m "feat(notebook): add test-split evaluation cell with mAP + per-class AP"
```

---

## Task 10: Add TFLite export cells (INT8 + FP16)

Spec section: *TFLite export (mobile deployment)*.

**Files:**
- Modify: `scripts/gen_train_notebook.py`
- Modify: `tests/test_notebook_generation.py`
- Modify: `notebooks/train_l1_yolo11s.ipynb`

- [ ] **Step 1: Extend tests**

Add to `TrainingCellTests`:

```python
    def test_tflite_export_cells_present(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("format=\"tflite\"", joined)
        self.assertIn("int8=True", joined)
        self.assertIn("half=True", joined)
        self.assertIn("EXPORTS_DIR", joined)
```

- [ ] **Step 2: Run, confirm failing**

Run: `python -m unittest tests.test_notebook_generation.TrainingCellTests.test_tflite_export_cells_present -v`
Expected: fail.

- [ ] **Step 3: Add export cells**

In `build_cells()`, append after the eval cell:

```python
    # Cell: TFLite INT8 export (primary mobile artifact)
    cells.append(markdown(
        "### TFLite export - INT8 (primary)\n\n"
        "Quantized to INT8 using calibration images sampled from the training "
        "set. ~6 MB; preferred for the mobile app."
    ))
    cells.append(code(
        r"""int8_artifact = best_model.export(
    format="tflite",
    int8=True,
    imgsz=IMG_SIZE,
    data=DATA_YAML,
)
print("INT8 TFLite ->", int8_artifact)
shutil.copy(int8_artifact, os.path.join(EXPORTS_DIR, "best_int8.tflite"))
print("Copied to", os.path.join(EXPORTS_DIR, "best_int8.tflite"))"""
    ))

    # Cell: TFLite FP16 export (fallback)
    cells.append(markdown(
        "### TFLite export - FP16 (fallback)\n\n"
        "Half-precision fallback (~22 MB). Use if INT8 calibration causes "
        "accuracy regressions on-device."
    ))
    cells.append(code(
        r"""fp16_artifact = best_model.export(
    format="tflite",
    half=True,
    imgsz=IMG_SIZE,
)
print("FP16 TFLite ->", fp16_artifact)
shutil.copy(fp16_artifact, os.path.join(EXPORTS_DIR, "best_float16.tflite"))
print("Copied to", os.path.join(EXPORTS_DIR, "best_float16.tflite"))"""
    ))
```

- [ ] **Step 4: Run all tests + regenerate**

Run: `python -m unittest discover tests -v && python scripts/gen_train_notebook.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_train_notebook.py tests/test_notebook_generation.py notebooks/train_l1_yolo11s.ipynb
git commit -m "feat(notebook): add INT8 + FP16 TFLite export cells"
```

---

## Task 11: Add TFLite inference smoke test cell

Spec section: *Inference smoke test*.

**Files:**
- Modify: `scripts/gen_train_notebook.py`
- Modify: `tests/test_notebook_generation.py`
- Modify: `notebooks/train_l1_yolo11s.ipynb`

- [ ] **Step 1: Extend test**

Add to `TrainingCellTests`:

```python
    def test_smoke_test_cell_uses_tflite(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("best_int8.tflite", joined)
        self.assertIn("smoke_test.png", joined)
        self.assertIn(".predict(", joined)
```

- [ ] **Step 2: Run, confirm failing**

Run: `python -m unittest tests.test_notebook_generation.TrainingCellTests.test_smoke_test_cell_uses_tflite -v`
Expected: fail.

- [ ] **Step 3: Add smoke test cell**

In `build_cells()`, append after the FP16 export cell:

```python
    # Cell: inference smoke test against the INT8 TFLite artifact
    cells.append(markdown(
        "### Inference smoke test\n\n"
        "Loads the INT8 TFLite (the actual deployment artifact), runs one "
        "prediction on a random test image, and saves an annotated preview."
    ))
    cells.append(code(
        r"""import random

INT8_TFLITE = os.path.join(EXPORTS_DIR, "best_int8.tflite")
test_imgs = sorted((Path(DATA_DIR) / "images" / "test").glob("*"))
if not test_imgs:
    raise RuntimeError("No test images found.")
sample = random.choice(test_imgs)
print("Smoke-testing on:", sample.name)

deploy_model = YOLO(INT8_TFLITE)
results = deploy_model.predict(
    source=str(sample),
    imgsz=IMG_SIZE,
    conf=0.25,
    save=False,
)
res = results[0]

preview_path = os.path.join(EXPORTS_DIR, "smoke_test.png")
res.save(filename=preview_path)
print("Saved annotated preview to", preview_path)

if res.boxes is None or len(res.boxes) == 0:
    print("(no detections above conf=0.25)")
else:
    for box in res.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"  {L1_CLASSES[cls]}  conf={conf:.3f}  xyxy={xyxy}")"""
    ))
```

- [ ] **Step 4: Run all tests + regenerate**

Run: `python -m unittest discover tests -v && python scripts/gen_train_notebook.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_train_notebook.py tests/test_notebook_generation.py notebooks/train_l1_yolo11s.ipynb
git commit -m "feat(notebook): add TFLite inference smoke test cell"
```

---

## Task 12: Delete old classifier notebook

Spec section: *Repo changes → Replaced*.

**Files:**
- Delete: `notebooks/train_l1_efficientnetb4.ipynb`

- [ ] **Step 1: Confirm the old notebook still exists**

Run: `ls notebooks/`
Expected output includes both `train_l1_efficientnetb4.ipynb` and `train_l1_yolo11s.ipynb`.

- [ ] **Step 2: Remove the old notebook with git**

Run: `git rm notebooks/train_l1_efficientnetb4.ipynb`
Expected: `rm 'notebooks/train_l1_efficientnetb4.ipynb'`

- [ ] **Step 3: Verify directory**

Run: `ls notebooks/`
Expected: only `train_l1_yolo11s.ipynb`.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove classifier notebook (superseded by YOLO11s detector)"
```

---

## Task 13: Update README.md

Spec section: *Repo changes → Replaced* (`README.md`).

**Files:**
- Modify (full rewrite): `README.md`

- [ ] **Step 1: Replace `README.md` end-to-end**

Overwrite `README.md` with:

````markdown
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
````

- [ ] **Step 2: Verify the file**

Run: `head -20 README.md`
Expected: new header content.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for YOLO11s detection pipeline"
```

---

## Task 14: Final verification end-to-end

**Files:** None modified.

- [ ] **Step 1: Run the full test suite**

Run: `python -m unittest discover tests -v`
Expected: all tests pass (helper unit tests + notebook generation tests).

- [ ] **Step 2: Regenerate the notebook and inspect cell count**

Run: `python scripts/gen_train_notebook.py`
Expected: `Wrote ...train_l1_yolo11s.ipynb`.

Then: `python -c "import json; d=json.load(open('notebooks/train_l1_yolo11s.ipynb')); print('cells:', len(d['cells'])); [print(i, c['cell_type'], ''.join(c['source']).split(chr(10))[0][:60]) for i, c in enumerate(d['cells'])]"`
Expected: ~21 cells; the printed first-line preview should walk through header → install → imports → drive/HF → config → helpers heading → helpers code → dataset heading → dataset code → validation heading → validation code → train heading → train code → eval heading → eval code → INT8 export heading → INT8 export code → FP16 export heading → FP16 export code → smoke test heading → smoke test code.

- [ ] **Step 3: Verify the notebook is valid JSON and every code cell parses**

This is what `tests/test_notebook_generation.py` already does, but as a final manual sanity check:

Run: `python -c "import ast, json; d=json.load(open('notebooks/train_l1_yolo11s.ipynb')); [ast.parse(''.join(c['source'])) for c in d['cells'] if c['cell_type']=='code' and not ''.join(c['source']).lstrip().startswith(('%%capture','%pip','!pip'))]; print('all code cells parse')"`
Expected: `all code cells parse`.

- [ ] **Step 4: Final git status check**

Run: `git status && git log --oneline -15`
Expected: working tree clean; the last ~13 commits walk through the implementation in order.

- [ ] **Step 5: No commit needed**

This task is verification only.

---

## Coverage check (vs spec)

| Spec section | Implemented in |
|---|---|
| Paths and config variables | Task 5 (Cell 5) |
| Dataset → Source | Tasks 5, 6 (README and dataset cell) |
| Dataset → Class list | Task 5 (Cell 5 `L1_CLASSES`) |
| Dataset → Extraction flow | Tasks 3 (helper), 6 (notebook cell) |
| Dataset → Label validation | Tasks 2 (helper), 7 (notebook cell) |
| Model and training | Task 8 |
| Resume after disconnect | Task 8 (resume-on-last.pt) |
| Evaluation | Task 9 |
| TFLite export INT8 | Task 10 |
| TFLite export FP16 | Task 10 |
| Inference smoke test | Task 11 |
| Repo changes → `gen_train_notebook.py` rewrite | Tasks 5, 6, 7, 8, 9, 10, 11 |
| Repo changes → delete old `.ipynb` | Task 12 |
| Repo changes → new `.ipynb` generated | Tasks 5+ (regenerated each task) |
| Repo changes → `README.md` | Task 13 |
| Repo changes → `requirements.txt` | Task 1 |
| Notebook cell outline (1–11) | Tasks 5–11 |
| Success criteria 1 (end-to-end run produces artifacts) | All tasks; verified manually on Colab |
| Success criteria 2 (mAP + per-class AP) | Task 9 |
| Success criteria 3 (validation rejects malformed `.txt`) | Tasks 2, 7 |
| Success criteria 4 (resume from `last.pt`) | Task 8 |
| Success criteria 5 (smoke test on INT8 TFLite) | Task 11 |
