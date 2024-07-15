import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

# Funzione per applicare il filtro passa-basso
def low_pass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def calculate_snr(signal, filtered_signal):
    noise = signal - filtered_signal
    signal_power = np.mean(filtered_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Funzione per analizzare lo spettro del segnale
def plot_frequency_response(data, fs):
    freqs, psd = plt.psd(data, NFFT=1024, Fs=fs, scale_by_freq=True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.show()

# Caricamento dei dati
imu_data = pd.read_csv('../data/imu_fingerprints.csv')
# Calcolo della magnitudine dell'accelerazione
imu_data['acc_mag'] = np.sqrt(imu_data['ACC_X']**2 + imu_data['ACC_Y']**2 + imu_data['ACC_Z']**2)

# Analisi dello spettro del segnale
fs = 31  # Frequenza di campionamento in Hz
plot_frequency_response(imu_data['acc_mag'], fs)

# Applicazione del filtro passa-basso
cutoff = 2.0  # Frequenza di taglio in Hz
order = 3  # Ordine del filtro
imu_data['acc_mag_filtered'] = low_pass_filter(imu_data['acc_mag'], cutoff, fs, order)

# Calculate SNR
snr = calculate_snr(imu_data['acc_mag'], imu_data['acc_mag_filtered'])
print(f"SNR: {snr:.2f} dB")

# Visualizzazione del segnale filtrato
plt.figure(figsize=(20, 16))
plt.plot(imu_data['TIMESTAMP'], imu_data['acc_mag'], label='Original Signal')
plt.plot(imu_data['TIMESTAMP'], imu_data['acc_mag_filtered'], label='Filtered Signal', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Magnitude of Acceleration')
plt.legend()
plt.show()