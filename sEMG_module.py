"""
emg_fatigue_module.py

Standalone, real-time EMG fatigue/cramp-risk signal processing for the
PowerThru forearm armsleeve. Deliberately NOT wired into the WESAD-trained
stress classifier as an input channel -- WESAD has no forearm (or any wrist)
EMG channel to train against, so there's no labeled data anywhere to teach
a neural net what forearm EMG means for stress or cramp risk. This module
instead computes a risk score directly from established EMG fatigue
signatures, with no dataset dependency:

  - RMS (amplitude) tends to RISE as a muscle fatigues (more motor units
    recruited / firing faster to maintain the same force output)
  - Median frequency of the EMG power spectrum tends to FALL as a muscle
    fatigues (slower-conducting fiber types dominate more as fast fibers
    tire)

Both trends are computed causally -- only ever using the current and past
windows, nothing from the future -- so this is safe to run live on-device.

WHAT YOU NEED TO DO before this is anything more than a reasonable-looking
placeholder:
  1. Confirm the real ADC sampling rate you're running the AD8232 EMG
     channel at (FS_EMG below is a common default for EMG work, not a
     measured value from your hardware -- update it).
  2. The thresholds in FatigueRiskTracker.risk_score() are NOT calibrated to
     real cramp events -- they're directionally reasonable placeholders.
     Getting real numbers requires a small pilot: log raw EMG through a
     session where you (or a pilot subject) either fatigue a forearm muscle
     to a real cramp/near-cramp, or do a matched non-fatiguing control, and
     see what RMS/median-frequency trajectories actually precede real
     cramp onset for your electrode placement and grip. Without that,
     treat any output from this module as a rough directional signal, not
     a validated risk probability.
  3. Once you have (2), the WESAD-based stress branch and this fatigue
     branch can be fused into one output -- simplest version is a rule
     ("flag high risk if either exceeds its threshold"), a learned fusion
     becomes possible once you have enough paired pilot sessions to train
     one.
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch
from collections import deque


# NOTE: placeholder -- replace with your AD8232 circuit's actual ADC rate.
# 1000 Hz is common for EMG hobbyist/embedded setups (Nyquist well above the
# ~20-450 Hz EMG band) but confirm this against your firmware config.
FS_EMG = 1000.0

EMG_BANDPASS_LOW = 20.0
EMG_BANDPASS_HIGH = 450.0

# Trend window: how many successive feature windows to look back over when
# computing the RMS/median-frequency slope. At WINDOW_SEC=60/STEP_SEC=30
# (matching the WESAD pipeline's windowing) this is a 5-minute look-back;
# adjust once you know how fast your own cramp events actually develop.
TREND_LOOKBACK = 10


def bandpass_emg(sig, fs=FS_EMG, low=EMG_BANDPASS_LOW, high=EMG_BANDPASS_HIGH, order=4):
    """Standard EMG conditioning band. filtfilt is zero-phase but non-causal
    WITHIN a single call -- fine here because it's applied to a window
    that's already entirely in the past by the time it's processed (the
    window itself doesn't reach into the future), same reasoning as the
    windowed R-peak detection from the WESAD pipeline."""
    nyq = fs / 2.0
    high = min(high, nyq - 5.0)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def emg_rms(window):
    """Root-mean-square amplitude of a (already bandpassed) EMG window."""
    return float(np.sqrt(np.mean(np.square(window))))


def emg_median_frequency(window, fs=FS_EMG):
    """Median frequency of the EMG power spectrum: the frequency below
    which half the total spectral power lies. Falls as a muscle fatigues."""
    nperseg = min(len(window), int(fs * 2))
    if nperseg < 8:
        return float("nan")
    freqs, psd = welch(window, fs=fs, nperseg=nperseg)
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    if total_power <= 0:
        return float("nan")
    median_idx = int(np.searchsorted(cumulative_power, total_power / 2.0))
    median_idx = min(median_idx, len(freqs) - 1)
    return float(freqs[median_idx])


def _trend_slope(values):
    """Least-squares slope of a short trailing sequence of feature values
    against window index (not wall-clock time -- windows are evenly
    spaced by construction). NaNs are dropped before fitting; returns NaN
    if fewer than 3 valid points remain."""
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 3:
        return float("nan")
    x = np.arange(len(values))[valid]
    y = values[valid]
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


class FatigueRiskTracker:
    """
    Streaming, causal fatigue-risk estimator. Call update(emg_window) once
    per new window (e.g. every STEP_SEC seconds, matching the WESAD
    pipeline's cadence) with the raw EMG samples for that window; it
    maintains a rolling history internally and returns the current risk
    estimate. Never looks at a window you haven't given it yet.
    """

    def __init__(self, fs=FS_EMG, lookback=TREND_LOOKBACK):
        self.fs = fs
        self.lookback = lookback
        self.rms_history = deque(maxlen=lookback)
        self.mdf_history = deque(maxlen=lookback)

    def update(self, emg_window):
        """emg_window: 1D array of raw EMG samples for the current window
        only (no future samples). Returns a dict with the raw features,
        their trend slopes, and a placeholder 0-1 risk_score -- see the
        module docstring about calibrating this against real pilot data
        before trusting the number itself."""
        filtered = bandpass_emg(emg_window, fs=self.fs)
        rms = emg_rms(filtered)
        mdf = emg_median_frequency(filtered, fs=self.fs)

        self.rms_history.append(rms)
        self.mdf_history.append(mdf)

        rms_slope = _trend_slope(self.rms_history)
        mdf_slope = _trend_slope(self.mdf_history)

        return {
            "rms": rms,
            "median_freq": mdf,
            "rms_slope": rms_slope,
            "mdf_slope": mdf_slope,
            "risk_score": self._risk_score(rms_slope, mdf_slope),
        }

    def _risk_score(self, rms_slope, mdf_slope):
        """PLACEHOLDER scoring: classic fatigue signature is rms_slope > 0
        (amplitude rising) AND mdf_slope < 0 (spectrum shifting down) at
        the same time. Returns a soft 0-1 score using how many trend
        windows are actually available yet (0 until self.lookback windows
        have been seen, since a 2-point "slope" isn't trustworthy) --
        NOT validated against real cramp events. Recalibrate the slope
        thresholds (currently arbitrary small positive/negative numbers)
        once you have real pilot recordings with known cramp onsets."""
        if len(self.rms_history) < min(3, self.lookback):
            return 0.0
        if np.isnan(rms_slope) or np.isnan(mdf_slope):
            return 0.0

        # crude confidence ramp: don't claim a strong score until the
        # rolling window is reasonably full
        confidence = min(1.0, len(self.rms_history) / float(self.lookback))

        rms_rising = rms_slope > 0
        mdf_falling = mdf_slope < 0
        if rms_rising and mdf_falling:
            return round(0.7 * confidence + 0.3, 3)
        elif rms_rising or mdf_falling:
            return round(0.3 * confidence, 3)
        else:
            return 0.0


if __name__ == "__main__":
    # Self-test with synthetic EMG: a "resting" segment followed by a
    # "fatiguing" segment (rising amplitude, energy shifting to lower
    # frequencies) -- just to sanity-check the trend direction is picked
    # up correctly. This is NOT a substitute for testing on real EMG.
    rng = np.random.RandomState(0)
    fs = FS_EMG
    window_len = int(fs * 2)  # 2s windows for this quick self-test

    tracker = FatigueRiskTracker(fs=fs, lookback=TREND_LOOKBACK)

    print("Resting windows (should stay low risk):")
    for i in range(5):
        t = np.arange(window_len) / fs
        resting = 0.3 * np.sin(2 * np.pi * 80 * t) * rng.randn(window_len) * 0.5
        result = tracker.update(resting)
        print(f"  window {i}: rms={result['rms']:.4f} mdf={result['median_freq']:.1f}Hz "
              f"risk={result['risk_score']}")

    print("Fatiguing windows (amplitude up, frequency down; risk should climb):")
    for i in range(10):
        t = np.arange(window_len) / fs
        amp = 0.3 + 0.05 * i
        freq_shift = max(40.0, 80.0 - 4.0 * i)  # spectral content drifting down
        fatiguing = amp * np.sin(2 * np.pi * freq_shift * t) * (1 + 0.3 * rng.randn(window_len))
        result = tracker.update(fatiguing)
        print(f"  window {i}: rms={result['rms']:.4f} mdf={result['median_freq']:.1f}Hz "
              f"risk={result['risk_score']}")
