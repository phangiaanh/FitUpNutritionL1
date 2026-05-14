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
