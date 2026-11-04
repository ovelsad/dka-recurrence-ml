"""Проверки расчетов и кэша на искусственных данных"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import config, evaluate, stats, synthetic, tuning


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]},
            index=[11, 22, 33, 44],
        )
        self.y = pd.Series([0, 1, 0, 1], index=self.x.index, name="outcome")
        self.oof = pd.DataFrame(
            {i: [0.1, 0.8, 0.3, 0.7] for i in range(evaluate.N_REPEATS)},
            index=self.x.index,
        )

    def test_ordered_fingerprint(self):
        original = tuning.cache_fingerprint(self.x, self.y)
        self.assertEqual(
            original, tuning.cache_fingerprint(self.x.copy(), self.y.copy())
        )
        self.assertNotEqual(
            original, tuning.cache_fingerprint(self.x.iloc[::-1], self.y.iloc[::-1])
        )
        self.assertNotEqual(
            original, tuning.cache_fingerprint(self.x[["b", "a"]], self.y)
        )
        changed = self.x.copy()
        changed.iloc[0, 0] = 99
        self.assertNotEqual(original, tuning.cache_fingerprint(changed, self.y))

    def test_fingerprint_rejects_misalignment(self):
        with self.assertRaises(ValueError):
            tuning.cache_fingerprint(self.x, self.y.iloc[::-1])

    def test_validate_oof(self):
        evaluate.validate_oof(self.oof, self.y)
        with self.assertRaises(ValueError):
            evaluate.validate_oof(self.oof.iloc[::-1], self.y)
        with self.assertRaises(ValueError):
            evaluate.validate_oof(self.oof.drop(columns=0), self.y)
        for value in [np.nan, np.inf, -0.1, 1.1]:
            broken = self.oof.copy()
            broken.iloc[0, 0] = value
            with self.assertRaises(ValueError):
                evaluate.validate_oof(broken, self.y)

    def test_summary_rejects_misalignment_before_metrics(self):
        with self.assertRaises(ValueError):
            evaluate.summarize({"oof": self.oof.iloc[::-1], "y": self.y})

    def test_tree_cache_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "CatBoost"
            directory.mkdir()
            info = {"настройки": tuning.settings("CatBoost", self.x, self.y)}
            (directory / "info.json").write_text(json.dumps(info), encoding="utf-8")
            self.oof.iloc[::-1].to_csv(directory / "oof.csv")
            with (
                patch.object(tuning, "CACHE_DIR", Path(temp)),
                self.assertRaises(ValueError),
            ):
                tuning.cross_validate_cached("CatBoost", self.x, self.y)

    def test_synthetic_cache_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "demo"
            directory.mkdir()
            (directory / "info.json").write_text(
                json.dumps({"настройки": {}}), encoding="utf-8"
            )
            self.oof.iloc[::-1].to_csv(directory / "oof.csv")
            with (
                patch.object(synthetic, "CACHE_DIR", Path(temp)),
                patch.object(synthetic, "settings", return_value={}),
                self.assertRaises(ValueError),
            ):
                synthetic.cross_validate_cached("demo", None, self.x, self.y)

    def test_split_is_disjoint_and_complete(self):
        with patch.object(config, "TEST_IDS", (22, 44)):
            train, test, y_train, y_test = evaluate.split(self.x, self.y)
        self.assertEqual(set(train.index), {11, 33})
        self.assertEqual(set(test.index), {22, 44})
        self.assertEqual(len(train) + len(test), len(self.x))
        self.assertTrue(train.index.equals(y_train.index))
        self.assertTrue(test.index.equals(y_test.index))

    def test_split_rejects_missing_test_patient(self):
        with patch.object(config, "TEST_IDS", (99,)), self.assertRaises(ValueError):
            evaluate.split(self.x, self.y)

    def test_youden_threshold_separates_known_example(self):
        self.assertAlmostEqual(
            evaluate.youden_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9]), 0.7
        )

    def test_odds_ratio_direction(self):
        value, low, high = stats.odds_ratio(12, 6, 8, 24)
        reverse, _, _ = stats.odds_ratio(8, 24, 12, 6)
        self.assertAlmostEqual(value, 6)
        self.assertAlmostEqual(reverse, 1 / 6)
        self.assertLess(low, value)
        self.assertGreater(high, value)


if __name__ == "__main__":
    unittest.main()
