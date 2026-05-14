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
            self.assertEqual(len(doc["cells"]), 13)

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

    def test_label_validation_cell_present(self) -> None:
        joined = "\n".join(self._sources())
        self.assertIn("validate_yolo_labels", joined)


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


if __name__ == "__main__":
    unittest.main()
