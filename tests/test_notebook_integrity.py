from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "RLHF_HW_fixed.ipynb"
README = ROOT / "README.md"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


class NotebookIntegrityTests(unittest.TestCase):
    def test_notebook_code_cells_are_valid_python_or_shell_setup(self) -> None:
        for index, cell in enumerate(_notebook()["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            has_shell_magic = any(
                line.lstrip().startswith(("!", "%")) for line in source.splitlines()
            )
            if has_shell_magic:
                continue
            ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{index}")

    def test_colab_badge_points_to_this_repository(self) -> None:
        first_cell = _notebook()["cells"][0]
        source = "".join(first_cell["source"])
        self.assertIn("FengyuanLiu1101/RLHF-make-review-positive", source)
        self.assertIn("RLHF_HW_fixed.ipynb", source)

    def test_readme_references_existing_notebook_and_requirements(self) -> None:
        readme = README.read_text(encoding="utf-8")
        self.assertIn("RLHF_HW_fixed.ipynb", readme)
        self.assertIn("requirements.txt", readme)
        self.assertTrue((ROOT / "requirements.txt").exists())


if __name__ == "__main__":
    unittest.main()
