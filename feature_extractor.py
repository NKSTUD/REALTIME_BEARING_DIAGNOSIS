import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def calculate_features(data):
    return {
        "crest_factor": np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data))),
        "kurtosis": pd.Series(data).kurtosis(),
        "skewness": pd.Series(data).skew(),
        "rms": np.sqrt(np.mean(np.square(data)))
    }


class Signal:
    """
    This class represents a signal with associated data, time, and origin information.
    It also includes methods for extracting various features from the signal.
    """

    def __init__(self, data, sampling_rate=None, origin: str = ""):
        self.data = np.array(data)
        self.origin = origin
        self.sampling_rate = sampling_rate if sampling_rate else len(self.data)

    def __str__(self):
        return f"This is a {self.origin} signal with a length of {len(self.data)}."

    def get_time(self):
        return np.linspace(0, len(self.data) / self.sampling_rate, len(self.data))

    def get_fft(self):
        return np.abs(np.fft.fft(self.data))

    def get_frequencies(self):
        time = self.get_time()
        return np.fft.fftfreq(len(time), time[1] - time[0])

    # Time-domain feature extraction methods
    def square_root_amplitude(self):
        return (np.mean(np.sqrt(np.abs(self.data)))) ** 2

    def absolute_average(self):
        return np.mean(np.abs(self.data))

    def rms(self):
        return np.sqrt(np.mean(np.square(self.data)))

    def amplitude_max(self):
        return np.max(self.data)

    def amplitude_min(self):
        return np.min(self.data)

    def variance(self):
        return np.var(self.data, ddof=1)

    def get_kurtosis(self):
        return kurtosis(self.data)

    def get_mean(self):
        return np.mean(self.data)

    def get_skewness(self):
        return skew(self.data)

    def peak_to_peak(self):
        return self.amplitude_max() - self.amplitude_min()

    def shape_factor(self):
        absolute_avg = self.absolute_average()
        return self.rms() / absolute_avg if absolute_avg != 0 else 0

    def crest_factor(self):
        rms_value = self.rms()
        return self.amplitude_max() / rms_value if rms_value != 0 else 0

    def impulse_factor(self):
        absolute_avg = self.absolute_average()
        return self.amplitude_max() / absolute_avg if absolute_avg != 0 else 0

    def coefficient_of_variation(self):
        square_root_amp = self.square_root_amplitude()
        return self.amplitude_max() / square_root_amp if square_root_amp != 0 else 0

    def coefficient_of_skewness(self):
        variance = self.variance()
        return self.get_skewness() / (np.sqrt(variance)) ** 3 if variance != 0 else 0

    def coefficient_of_kurtosis(self):
        variance = self.variance()
        return self.get_kurtosis() / (np.sqrt(variance)) ** 4 if variance != 0 else 0

    # Frequency-domain feature extraction methods
    def mean_frequency(self):
        return np.mean(self.get_fft())

    def spectral_variance(self):
        return np.var(self.get_fft())

    def spectral_skewness(self):
        return skew(np.abs(self.get_fft()))

    def spectral_kurtosis(self):
        return kurtosis(np.abs(self.get_fft()))

    def frequency_center(self):
        fft = self.get_fft()
        frequencies = self.get_frequencies()
        total_power = np.sum(np.abs(fft))
        return np.sum(np.abs(fft) * frequencies) / total_power if total_power != 0 else 0

    def standard_deviation_frequency(self):
        mean_freq = self.frequency_center()
        fft = self.get_fft()
        frequencies = self.get_frequencies()
        total_power = np.sum(np.abs(fft))
        return np.sqrt(np.sum((frequencies - mean_freq) ** 2 * np.abs(fft)) / total_power) if total_power != 0 else 0

    def rms_frequency(self):
        fft = self.get_fft()
        frequencies = self.get_frequencies()
        total_power = np.sum(np.abs(fft))
        return np.sqrt(np.sum(frequencies ** 2 * np.abs(fft)) / total_power) if total_power != 0 else 0

    def extract_features(self):
        features = {
            "kurtosis": self.get_kurtosis(),
            "skewness": self.get_skewness(),
            "amplitude_max": self.amplitude_max(),
            "amplitude_min": self.amplitude_min(),
            "rms": self.rms(),
            "mean": self.get_mean(),
            "absolute_average": self.absolute_average(),
            "variance": self.variance(),
            "square_root_amplitude": self.square_root_amplitude(),
            "peak_to_peak": self.peak_to_peak(),
            "shape_factor": self.shape_factor(),
            "crest_factor": self.crest_factor(),
            "impulse_factor": self.impulse_factor(),
            "coefficient_of_variation": self.coefficient_of_variation(),
            "coefficient_of_skewness": self.coefficient_of_skewness(),
            "coefficient_of_kurtosis": self.coefficient_of_kurtosis(),
            "mean_frequency": self.mean_frequency(),
            "spectral_variance": self.spectral_variance(),
            "spectral_skewness": self.spectral_skewness(),
            "spectral_kurtosis": self.spectral_kurtosis(),
            "frequency_center": self.frequency_center(),
            "standard_deviation_frequency": self.standard_deviation_frequency(),
            "rms_frequency": self.rms_frequency()
        }
        return features


all_features = [
    "kurtosis", "skewness", "amplitude_max", "amplitude_min", "rms", "mean",
    "absolute_average", "variance", "square_root_amplitude", "peak_to_peak",
    "shape_factor", "crest_factor", "impulse_factor", "coef_variation",
    "coef_skewness", "coef_kurtosis", "mean_frequency", "spectral_variance",
    "spectral_skewness", "spectral_kurtosis", "frequency_center",
    "standard_deviation_frequency", "rms_freq"
]

model_features = [
    "kurtosis",
    "skewness",
    "amplitude_max",
    "rms",
    "mean",
    "absolute_average",
    "peak_to_peak",
    "shape_factor",
    "crest_factor",
    "impulse_factor",
    "spectral_kurtosis",
    "frequency_center",
    "rms_freq"
]


def get_time(data, sampling_rate=25600):
    return np.linspace(0, len(data) / sampling_rate, len(data))


def getfft(data):
    return np.abs(np.fft.fft(data))


def getfrequencies(data):
    time = get_time(data)
    return np.fft.fftfreq(len(time), time[1] - time[0])


def extract_model_features(data):
    fft = getfft(data)
    return {
        "kurtosis": kurtosis(data),
        "skewness": skew(data),
        "amplitude_max": np.max(data),
        "rms": np.sqrt(np.mean(np.square(data))),
        "mean": np.mean(data),
        "absolute_average": np.mean(np.abs(data)),
        "peak_to_peak": np.ptp(data),
        "shape_factor": np.sqrt(np.mean(np.square(data))) / np.mean(np.abs(data)),
        "crest_factor": np.max(data) / np.sqrt(np.mean(np.square(data))),
        "impulse_factor": np.max(data) / np.mean(np.abs(data)),
        "mean_frequency": np.mean(fft),
        "spectral_kurtosis": kurtosis(np.abs(fft)),
        "frequency_center": np.sum(np.abs(fft) * getfrequencies(data)) / np.sum(np.abs(fft)),
        "rms_freq": np.sqrt(np.sum((getfrequencies(data)) ** 2 * np.abs(fft)) / np.sum(np.abs(fft)))
    }


def extract_all_features(data):
    fft = getfft(data)

    return extract_model_features(data) | {
        "variance": np.var(data),
        "square_root_amplitude": (np.mean(np.sqrt(np.abs(data)))) ** 2,
        "amplitude_min": np.min(data),
        "coef_variation": np.max(data) / (np.mean(np.sqrt(np.abs(data)))) ** 2,
        "coef_skewness": skew(data) / (np.sqrt(np.var(data))) ** 3,
        "coef_kurtosis": kurtosis(data) / (np.sqrt(np.var(data))) ** 4,
        "spectral_variance": np.var(fft),
        "spectral_skewness": skew(np.abs(fft)),
        "standard_deviation_frequency": np.sqrt(
            np.sum((getfrequencies(data) - np.mean(fft)) ** 2 * np.abs(fft)) / np.sum(np.abs(fft))
        ) if np.sum(np.abs(fft)) != 0 else 0

    }
