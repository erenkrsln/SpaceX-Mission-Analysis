import unittest
import pandas as pd
from src.analysis import clean_launches, summarize, evaluate_model

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "flight_number": [1, 2, 3],
            "date_utc": ["2020-01-01", "2020-05-01", "2021-01-01"],
            "success": [True, False, True],
            "reused": [False, True, True],
            "attempts": [0, 1, 2],
        })

    def test_summary(self):
        summary = summarize(clean_launches(self.frame))
        self.assertEqual(summary["launches"].tolist(), [2, 1])
        self.assertEqual(summary["success_rate"].tolist(), [0.5, 1.0])

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            clean_launches(self.frame.drop(columns=["success"]))

    def test_insufficient_data(self):
        self.assertNotEqual(evaluate_model(clean_launches(self.frame))["status"], "ok")

if __name__ == "__main__":
    unittest.main()
