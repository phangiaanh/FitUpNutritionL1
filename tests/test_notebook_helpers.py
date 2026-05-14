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


    def test_raises_on_missing_split_dir(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            # only create train, not val/test
            _make_split(root, "train", [("a", "0 0.5 0.5 0.5 0.5\n")])
            with self.assertRaisesRegex(ValueError, "missing image dir"):
                validate_yolo_labels(root, CLASSES)


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


if __name__ == "__main__":
    unittest.main()
