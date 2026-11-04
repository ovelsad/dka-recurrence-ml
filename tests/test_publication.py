"""Проверяет, что упаковка отсекает непроверенные материалы"""

import hashlib
import json
import unittest

from scripts.check_publication import inspect_content


class PublicationTests(unittest.TestCase):
    def test_private_file_rejected_even_if_listed(self):
        self.assertTrue(inspect_content("data/raw/example.xlsx", b"example", {}))
        self.assertTrue(inspect_content("other/model.pkl", b"example", {}))
        self.assertTrue(inspect_content("../outside.txt", b"example", {}))

    def test_clean_notebook_and_outputs(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["1 + 1"],
                    "outputs": [],
                    "execution_count": None,
                    "metadata": {},
                }
            ]
        }
        clean = json.dumps(notebook).encode()
        self.assertFalse(inspect_content("notebooks/demo.ipynb", clean, {}))
        notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": "example"}]
        self.assertTrue(
            inspect_content("notebooks/demo.ipynb", json.dumps(notebook).encode(), {})
        )

    def test_csv_requires_review_and_exact_content(self):
        content = b"model,score\na,0.7\n"
        entry = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "columns": ["model", "score"],
            "rows": 1,
        }
        self.assertFalse(
            inspect_content("supplementary/tables/demo.csv", content, entry)
        )
        self.assertTrue(
            inspect_content(
                "supplementary/tables/demo.csv", content.replace(b"0.7", b"0.8"), entry
            )
        )
        self.assertTrue(inspect_content("other/demo.csv", content, entry))

    def test_patient_identifier_column_rejected(self):
        content = b"patient_id,score\n1,0.7\n"
        entry = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "columns": ["patient_id", "score"],
            "rows": 1,
        }
        self.assertTrue(
            inspect_content("supplementary/tables/demo.csv", content, entry)
        )

    def test_unreviewed_binary_rejected(self):
        self.assertTrue(inspect_content("paper/demo.pdf", b"example", {}))

    def test_widget_state_rejected_without_outputs(self):
        notebook = {"cells": [], "metadata": {"widgets": {"state": "private example"}}}
        self.assertTrue(
            inspect_content("notebooks/demo.ipynb", json.dumps(notebook).encode(), {})
        )


if __name__ == "__main__":
    unittest.main()
