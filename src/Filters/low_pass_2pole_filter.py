import unittest
import os
import math
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def super_smoother(price_series, period):
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period) # FIXED: requires radians
    c1 = 1 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1

    smooth_series = [0] * len(price_series)
    smooth_series[0] = price_series[0]
    smooth_series[1] = price_series[1]

    for i in range(2, len(price_series)):
        smooth_series[i] = (
            c1 * (price_series[i] + price_series[i - 1]) / 2
            + c2 * smooth_series[i - 1]
            + c3 * smooth_series[i - 2]
        )
    return smooth_series


"""
#Usage example:
symbol = 'ES=F'
fetcher = DataFetcher(symbol=symbol)
price_data = fetcher.get_stock_data()
smoothed_prices = super_smoother(price_data.squeeze().tolist(), 14)
smoothed_df = pd.DataFrame(smoothed_prices, index=price_data.index, columns=['Smoothed'])
print(smoothed_df[:10])
"""

class TestSuperSmoother(unittest.TestCase):
    def test_constant_input(self):
        """
        For a constant input, the low-pass filter should return the same constant.
        """
        period = 20
        constant_series = [5] * 100  # DC input signal.
        smooth = super_smoother(constant_series, period)
        # The output should be nearly identical to the input.
        self.assertTrue(np.allclose(smooth, constant_series, atol=1e-6),
                        msg="Low-pass filter did not pass the constant input unchanged.")

    @unittest.skipUnless(os.getenv("RUN_INTERACTIVE_FREQ_TESTS") == "true", "Skipping frequency domain interactive plots test")
    def test_frequency_domain_plots(self):
        """
        Computes and plots four frequency responses for the low-pass filter in a single figure:
          1. Normalized Frequency Response (Magnitude in dB vs. normalized frequency)
          2. Magnitude Frequency Response (Magnitude in dB vs. actual frequency, cycles per day)
          3. Phase Frequency Response (Phase in radians vs. actual frequency, cycles per day)
          4. Group Delay (samples) vs. actual frequency (cycles per day)
        
        Also asserts that the DC gain is near 1.
        """
        period = 20
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
        c1 = 1 - b1 + a1 * a1
        c2 = b1
        c3 = -a1 * a1

        # The super_smoother recursion:
        #   y[i] = c1*(x[i] + x[i-1])/2 + c2*y[i-1] + c3*y[i-2]
        # corresponds to an IIR filter with:
        #   Numerator (b): [c1/2, c1/2, 0]
        #   Denominator (a): [1, -c2, -c3]
        b_coeffs = [c1 / 2, c1 / 2, 0]
        a_coeffs = [1, -c2, -c3]

        # Compute the frequency response.
        w, h = signal.freqz(b_coeffs, a_coeffs, worN=8000)
        freqs_normalized = w / np.pi  # Normalized frequency (0 to 1)

        # Assume a sampling frequency of 1 sample per day.
        Fs = 1.0
        nyquist = Fs / 2.0  # Nyquist frequency in cycles per day.
        actual_freq = freqs_normalized * nyquist

        # Compute group delay on a defined frequency grid.
        w_gd = np.linspace(0, np.pi, 8000)
        w_gd, gd = signal.group_delay((b_coeffs, a_coeffs), w=w_gd)
        freqs_normalized_gd = w_gd / np.pi
        actual_freq_gd = freqs_normalized_gd * nyquist

        # Use an epsilon to avoid log(0) issues.
        epsilon = 1e-12
        mag_response_db = 20 * np.log10(np.maximum(np.abs(h), epsilon))

        # Create a single figure with four vertically stacked subplots.
        plt.figure(figsize=(12, 16))

        # Plot 1: Normalized Frequency Response (Magnitude)
        plt.subplot(4, 1, 1)
        plt.plot(freqs_normalized, mag_response_db, 'b')
        plt.title("Normalized Frequency Response (Low-Pass)")
        plt.xlabel("Normalized Frequency (×π rad/sample)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)

        # Plot 2: Magnitude Frequency Response (Actual Frequency)
        plt.subplot(4, 1, 2)
        plt.plot(actual_freq, mag_response_db, 'r')
        plt.title("Magnitude Frequency Response (Low-Pass)")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)

        # Plot 3: Phase Frequency Response (Actual Frequency)
        plt.subplot(4, 1, 3)
        plt.plot(actual_freq, np.angle(h), 'g')
        plt.title("Phase Frequency Response (Low-Pass)")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Phase (radians)")
        plt.grid(True)

        # Plot 4: Group Delay (Actual Frequency)
        plt.subplot(4, 1, 4)
        plt.plot(actual_freq_gd, gd, 'm')
        plt.title("Group Delay (Low-Pass)")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Group Delay (samples)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Assert that the DC gain is near 1 (0 dB).
        dc_gain = np.abs(h[0])
        self.assertAlmostEqual(dc_gain, 1, delta=1e-2,
                               msg="DC gain should be near 1 for a low-pass filter.")

if __name__ == '__main__':
    unittest.main()
