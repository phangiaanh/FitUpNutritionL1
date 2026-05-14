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
                    ) from None
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
                    ) from None
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
