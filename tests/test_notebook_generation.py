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
