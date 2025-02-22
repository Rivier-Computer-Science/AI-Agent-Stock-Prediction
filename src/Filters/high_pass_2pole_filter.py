import unittest
import os
import math
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#import plotly.graph_objects as go


def highpass_2pole_filter(price_series, period):
    """
    Implements a highpass filter.
    Args:
        price_series (list): A time series of price data.
        period (int): Period of the filter.
    Returns:
        list: Highpass filtered series.
    """
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period) # FIXED: cos in python is in radians
    c1 = (1 + b1) / 4                                #  this moved the 3dB point to the left passing more frequencies
    c2 = b1
    c3 = -a1 * a1

    highpass_series = [0] * len(price_series)

    for i in range(2, len(price_series)):
        highpass_series[i] = (
            c1 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2])
            + c2 * highpass_series[i - 1]
            + c3 * highpass_series[i - 2]
        )
    return highpass_series


class TestHighpassFilter(unittest.TestCase):
    def test_constant_input(self):
        """For a constant input, the high-pass filter should eventually yield zero output."""
        constant_series = [5] * 100  # A constant (DC) signal
        period = 20
        filtered = highpass_2pole_filter(constant_series, period)
        
        # Skip initial transient period; expect near-zero output for later samples.
        self.assertTrue(np.allclose(filtered[10:], 0, atol=1e-6),
                        msg="The filter did not remove the DC component as expected.")

    @unittest.skipUnless(os.getenv("RUN_INTERACTIVE_FREQ_TESTS") == "true", "Skipping frequency domain interactive plots test")
    def test_frequency_domain_plots(self):
        """
        Computes and plots four frequency responses for the high-pass filter in a single figure:
          1. Normalized Frequency Response: Magnitude (dB) vs. normalized frequency.
          2. Magnitude Frequency Response: Magnitude (dB) vs. actual frequency (cycles per day).
          3. Phase Frequency Response: Phase (radians) vs. actual frequency (cycles per day).
          4. Group Delay: Group delay (samples) vs. actual frequency (cycles per day).
        
        Also asserts that the DC gain is near 0.
        """
        period = 20
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
        c1 = (1 + b1) / 4
        c2 = b1
        c3 = -a1 * a1

        # Transfer function coefficients for the high-pass filter:
        # y[i] = c1*(x[i] - 2*x[i-1] + x[i-2]) + c2*y[i-1] + c3*y[i-2]
        # Numerator coefficients:
        b_coeffs = [c1, -2 * c1, c1]
        # Denominator coefficients:
        a_coeffs = [1, -c2, -c3]

        # Compute the frequency response.
        w, h = signal.freqz(b_coeffs, a_coeffs, worN=8000)
        freqs_normalized = w / np.pi  # Normalized frequency (0 to 1)

        # Assume a sampling frequency of 1 sample per day.
        Fs = 1.0
        nyquist = Fs / 2.0  # Nyquist frequency in cycles per day.
        actual_freq = freqs_normalized * nyquist

        # Compute group delay using a frequency grid with the 'w' parameter.
        w_gd = np.linspace(0, np.pi, 8000)
        w_gd, gd = signal.group_delay((b_coeffs, a_coeffs), w=w_gd)
        freqs_normalized_gd = w_gd / np.pi
        actual_freq_gd = freqs_normalized_gd * nyquist

        # Use an epsilon to avoid log(0) issues.
        epsilon = 1e-12
        mag_response_db = 20 * np.log10(np.maximum(np.abs(h), epsilon))

        # Create a single figure with four subplots (stacked vertically)
        plt.figure(figsize=(12, 16))

        # Plot 1: Normalized Frequency Response (Magnitude)
        plt.subplot(4, 1, 1)
        plt.plot(freqs_normalized, mag_response_db, 'b')
        plt.title("Normalized Frequency Response (High-Pass)")
        plt.xlabel("Normalized Frequency (×π rad/sample)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)

        # Plot 2: Magnitude Frequency Response (Actual Frequency)
        plt.subplot(4, 1, 2)
        plt.plot(actual_freq, mag_response_db, 'r')
        plt.title("Magnitude Frequency Response (High-Pass)")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)

        # Plot 3: Phase Frequency Response (Actual Frequency)
        plt.subplot(4, 1, 3)
        plt.plot(actual_freq, np.angle(h), 'g')
        plt.title("Phase Frequency Response (High-Pass)")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Phase (radians)")
        plt.grid(True)

        # Plot 4: Group Delay (Actual Frequency)
        plt.subplot(4, 1, 4)
        plt.plot(actual_freq_gd, gd, 'm')
        plt.title("Group Delay (High-Pass)")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Group Delay (samples)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # For a high-pass filter, the DC gain should be near 0.
        dc_gain = np.abs(h[0])
        self.assertAlmostEqual(dc_gain, 0, delta=1e-2,
                               msg="DC gain should be near 0 for a high-pass filter.")



if __name__ == '__main__':
    unittest.main()