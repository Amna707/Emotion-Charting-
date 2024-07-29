
import scipy.io
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import pywt

def load_amigos_data(file_path):
    """
    Load EEG, ECG, and GSR data from a given AMIGOS dataset .mat file.
    
    :param file_path: str, path to the .mat file
    :return: dict containing EEG, ECG, and GSR data
    """
    data = scipy.io.loadmat(file_path)

    # Assuming the structure of the .mat file is known and it contains 'data' key
    eeg_data = data['data']['EEG'][0,0]
    ecg_data = data['data']['ECG'][0,0]
    gsr_data = data['data']['GSR'][0,0]
    
    return {
        'EEG': eeg_data,
        'ECG': ecg_data,
        'GSR': gsr_data
    }

def convert_to_dataframe(signal_data):
    """
    Convert the signal data to a pandas DataFrame.
    
    :param signal_data: numpy array of signal data
    :return: pandas DataFrame
    """
    df = pd.DataFrame(signal_data)
    return df

def segment_signal(signal_data, window_size):
    """
    Segment the signal data into non-overlapping windows of a given size.
    
    :param signal_data: numpy array of signal data
    :param window_size: int, number of samples per window
    :return: list of segmented signal data as numpy arrays
    """
    num_segments = len(signal_data) // window_size
    segmented_data = []
    for i in range(num_segments):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        segmented_data.append(signal_data[start_idx:end_idx])
    
    return segmented_data

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the signal.
    
    :param data: numpy array of signal data
    :param lowcut: float, lower frequency bound
    :param highcut: float, upper frequency bound
    :param fs: float, sampling frequency
    :param order: int, order of the filter
    :return: numpy array of filtered signal data
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def apply_ica(signal_data, n_components=None):
    """
    Apply Independent Component Analysis (ICA) to the signal data.
    
    :param signal_data: numpy array of signal data
    :param n_components: int, number of components to use for ICA
    :return: numpy array of ICA transformed data
    """
    ica = FastICA(n_components=n_components)
    transformed_data = ica.fit_transform(signal_data)
    return transformed_data

def apply_wavelet_transform(signal_data, wavelet='db4', level=4):
    """
    Apply Wavelet Transform to the signal data.
    
    :param signal_data: numpy array of signal data
    :param wavelet: str, name of the wavelet to use
    :param level: int, decomposition level
    :return: list of numpy arrays, wavelet coefficients
    """
    coeffs = pywt.wavedec(signal_data, wavelet, level=level, axis=0)
    return coeffs

# Example usage
file_path = 'path_to_your_file.mat'  # Replace with your file path
signals = load_amigos_data(file_path)

# Define the window size for 30 seconds at 128 Hz
window_size = 30 * 128  # 3840 samples

# Bandpass filter parameters
eeg_lowcut, eeg_highcut = 1.0, 50.0
ecg_lowcut, ecg_highcut = 0.5, 40.0
gsr_lowcut, gsr_highcut = 0.01, 5.0
sampling_frequency = 128  # 128 Hz

# Apply bandpass filter to each signal
filtered_eeg = bandpass_filter(signals['EEG'], eeg_lowcut, eeg_highcut, sampling_frequency)
filtered_ecg = bandpass_filter(signals['ECG'], ecg_lowcut, ecg_highcut, sampling_frequency)
filtered_gsr = bandpass_filter(signals['GSR'], gsr_lowcut, gsr_highcut, sampling_frequency)

# Segment each filtered signal
eeg_segments = segment_signal(filtered_eeg, window_size)
ecg_segments = segment_signal(filtered_ecg, window_size)
gsr_segments = segment_signal(filtered_gsr, window_size)

# Apply ICA to each segment (example: applying to the first segment)
ica_eeg = apply_ica(eeg_segments[0])
ica_ecg = apply_ica(ecg_segments[0])
ica_gsr = apply_ica(gsr_segments[0])

# Apply Wavelet Transform to each ICA component (example: applying to the first segment)
wavelet_eeg = apply_wavelet_transform(ica_eeg)
wavelet_ecg = apply_wavelet_transform(ica_ecg)
wavelet_gsr = apply_wavelet_transform(ica_gsr)

# Display the wavelet coefficients of the first ICA component
print("EEG Wavelet Coefficients of ICA Component 1:")
print(wavelet_eeg)
print("\nECG Wavelet Coefficients of ICA Component 1:")
print(wavelet_ecg)
print("\nGSR Wavelet Coefficients of ICA Component 1:")
print(wavelet_gsr)
