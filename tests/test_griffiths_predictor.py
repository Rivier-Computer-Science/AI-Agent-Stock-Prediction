# python -m pytest tests/test_griffiths_predictor.py 

import unittest
import numpy as np

from src.Indicators.griffiths_predictor import griffiths_predictor

class TestGriffithsPredictor(unittest.TestCase):
    def test_constant_input(self):
        """
        When the close prices are constant the filters should remove the DC component.
        Thus, the adaptive predictor should output near‑zero predictions and future signals.
        """
        # Create an array of constant prices.
        close_prices = np.full(100, 10.0)
        # Use default predictor parameters.
        predictions, future_signals = griffiths_predictor(close_prices, length=18, lower_bound=18, upper_bound=40, bars_fwd=2)
        
        # Since the input is constant, after the initial transient (first 'length' samples)
        # the predictor should have converged to a near‑zero output.
        self.assertTrue(np.allclose(predictions[18:], 0, atol=1e-6),
                        "For a constant input, predictions should be near zero after the initial transient.")
        self.assertTrue(np.allclose(future_signals, 0, atol=1e-6),
                        "For a constant input, future signals should be near zero.")

    def test_output_shape(self):
        """
        The predictions array should have the same shape as the input series.
        The future signals array should have a length equal to bars_fwd.
        """
        close_prices = np.linspace(1, 100, 100)  # A ramp signal.
        predictions, future_signals = griffiths_predictor(close_prices, length=20, lower_bound=15, upper_bound=30, bars_fwd=3)
        
        self.assertEqual(predictions.shape, close_prices.shape,
                         "Predictions should have the same shape as the input close_prices.")
        self.assertEqual(future_signals.shape[0], 3,
                         "Future signals should have length equal to bars_fwd (3 in this case).")

    def test_non_constant_input(self):
        """
        For a non-constant input (a sine wave in this example) the predictor should return finite values.
        """
        t = np.linspace(0, 2*np.pi, 100)
        close_prices = 10 + np.sin(t)  # Sine wave oscillating around 10.
        predictions, future_signals = griffiths_predictor(close_prices, length=18, lower_bound=18, upper_bound=40, bars_fwd=2)
        
        self.assertTrue(np.all(np.isfinite(predictions)),
                        "Predictions should contain finite numbers for a sine wave input.")
        self.assertTrue(np.all(np.isfinite(future_signals)),
                        "Future signals should contain finite numbers for a sine wave input.")

if __name__ == '__main__':
    unittest.main()
