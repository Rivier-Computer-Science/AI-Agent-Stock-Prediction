import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.Indicators.Griffiths_predictor import griffiths_predictor
from src.Indicators.High_pass_filter_function import highpass_filter
from src.Indicators.SuperSmoother_filter_function import super_smoother
from src.Data_Retrieval.data_fetcher import DataFetcher

class TestGriffithsPredictor(unittest.TestCase):
    def setUp(self):
        # Provide sample close prices for your predictor
        self.close_prices = [100, 101, 103, 102, 104, 107, 110, 108, 111, 115]
        self.length = 18
        self.lower_bound = 18
        self.upper_bound = 40
        self.bars_fwd = 2

    def test_griffiths_predictor_basic(self):
        """
        Test the core Griffiths predictor function returns the correct shape for predictions.
        """
        predictions, future_signals = griffiths_predictor(
            close_prices=self.close_prices,
            length=self.length,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            bars_fwd=self.bars_fwd
        )
        # The predictions array should be the same length as close_prices
        self.assertEqual(len(predictions), len(self.close_prices))
        # future_signals should have 'bars_fwd' length
        self.assertEqual(len(future_signals), self.bars_fwd)

    def test_griffiths_predictor_insufficient_data(self):
        """
        Test how the predictor handles very short input data.
        """
        short_data = [100, 101]  # Not enough bars
        with self.assertRaises(ValueError):
            # Suppose griffiths_predictor raises ValueError for insufficient data
            griffiths_predictor(
                close_prices=short_data,
                length=self.length,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                bars_fwd=self.bars_fwd
            )

    def test_highpass_filter_integration(self):
        """
        Test if the highpass_filter handles the same data cleanly.
        """
        result = highpass_filter(self.close_prices, period=20)  # Example period
        self.assertEqual(len(result), len(self.close_prices))
        # Maybe check that result isn't all zeros if there's enough data
        self.assertFalse(all(val == 0 for val in result[2:]))

    def test_supersmoother_integration(self):
        """
        Test if the supersmoother filter works with the same data.
        """
        result = super_smoother(self.close_prices, period=14)
        self.assertEqual(len(result), len(self.close_prices))
        self.assertFalse(all(val == 0 for val in result[2:]))


# ----------------------------
# Integration Test Cases
# ----------------------------

class TestGriffithsPredictorIntegration(unittest.TestCase):
    @patch('src.Data_Retrieval.data_fetcher.DataFetcher.get_stock_data')
    def test_griffiths_predictor_end_to_end(self, mock_stock_data):
        """
        Integration test that simulates fetching real data, then applying the Griffiths predictor.
        """
        # Mock out the get_stock_data return
        mock_stock_data.return_value = pd.DataFrame({
            'Close': [100, 101, 103, 102, 104, 107, 110, 108, 111, 115]
        })
        fetcher = DataFetcher()
        df = fetcher.get_stock_data("AAPL")

        # Convert to list for the predictor
        close_prices = df['Close'].tolist()

        # Now call the predictor
        predictions, future_signals = griffiths_predictor(
            close_prices=close_prices,
            length=18,
            lower_bound=18,
            upper_bound=40,
            bars_fwd=2
        )
        # Just basic checks for integration
        self.assertEqual(len(predictions), len(close_prices))
        self.assertEqual(len(future_signals), 2)

    @patch('src.Indicators.Griffiths_predictor.griffiths_predictor')
    def test_griffiths_predictor_with_market_scenario(self, mock_predictor):
        """
        Integration test to simulate a specific scenario, patching the actual predictor
        to return controlled output.
        """
        mock_predictor.return_value = (
            [101, 102, 103, 104],  # predictions
            [105, 106]            # future signals
        )
        # Suppose you have a class that uses the Griffiths predictor end-to-end:
        # (Pseudo example)
        # from src.SomeModule import GriffithsEngine
        # engine = GriffithsEngine()
        # result = engine.run("AAPL")
        # Check result

        self.assertTrue(mock_predictor.called)
        # self.assertIn("some text", result)

if __name__ == '__main__':
    unittest.main()
