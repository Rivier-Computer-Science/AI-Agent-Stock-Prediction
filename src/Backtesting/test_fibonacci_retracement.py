import unittest
from unittest.mock import patch
import pandas as pd
from src.Indicators.fibonacci import FibonacciRetracement
from src.Agents.Analysis.stock_analysis_tasks import StockAnalysisTasks
from src.Agents.Analysis.stock_analysis_agents import StockAnalysisAgents
from src.UI.main2 import FinancialCrew


# Unit Tests
class TestFibonacciRetracement(unittest.TestCase):

    def setUp(self):
        self.valid_data = pd.DataFrame({
            'High': [120, 130, 140],
            'Low': [100, 110, 115]
        })
        self.invalid_data = pd.DataFrame({
            'High': [120, 110],
            'Low': [130, 120]
        })
        self.fibonacci = FibonacciRetracement(self.valid_data)

    def test_invalid_high_low(self):
        fib = FibonacciRetracement(self.invalid_data)
        with self.assertRaises(ValueError):
            fib.calculate_levels()

    def test_identify_retracement_pattern(self):
        levels = self.fibonacci.calculate_levels()
        self.assertIn('61.8%', levels)

    def test_no_retracement_pattern(self):
        data = pd.DataFrame({'High': [120], 'Low': [100]})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertEqual(len(levels), 6)

    def test_negative_prices(self):
        data = pd.DataFrame({'High': [120, -130], 'Low': [100, -110]})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertIn('61.8%', levels)

    def test_identical_high_low(self):
        data = pd.DataFrame({'High': [120, 120], 'Low': [120, 120]})
        fib = FibonacciRetracement(data)
        with self.assertRaises(ValueError):
            fib.calculate_levels()

    def test_pattern_with_noise(self):
        data = pd.DataFrame({'High': [120, 130, 140, 125], 'Low': [100, 110, 115, 105]})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertIn('50%', levels)

    def test_sparse_data(self):
        data = pd.DataFrame({'High': [120], 'Low': [100]})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertEqual(len(levels), 6)

    def test_large_data_set(self):
        data = pd.DataFrame({'High': [120] * 1000, 'Low': [100] * 1000})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertEqual(len(levels), 6)

    def test_empty_data(self):
        data = pd.DataFrame(columns=['High', 'Low'])
        fib = FibonacciRetracement(data)
        with self.assertRaises(ValueError):
            fib.calculate_levels()

    def test_fibonacci_retracement_with_descending_prices(self):
        data = pd.DataFrame({'High': [140, 130, 120], 'Low': [110, 105, 100]})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertIn('61.8%', levels)

    def test_fibonacci_retracement_with_historical_comparison(self):
        data = pd.DataFrame({'High': [140, 130, 120], 'Low': [100, 90, 80]})
        fib = FibonacciRetracement(data)
        levels = fib.calculate_levels()
        self.assertIn('38.2%', levels)


if __name__ == '__main__':
    unittest.main()