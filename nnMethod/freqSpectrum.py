import numpy as np
import librosa
import librosa.display
from scipy.signal import find_peaks

def extract_frequencies(file_path, n_fft=2048, hop_length=512, peak_threshold=0.05, num_top_frequencies=6):
    """
    Extract top frequencies and their magnitudes from a WAV file.
    """
    y, sr = librosa.load(file_path, sr=None)
    
    # Compute the STFT and magnitude spectrum
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    sum_spectrum = np.sum(S, axis=1)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Find peaks
    peaks, _ = find_peaks(sum_spectrum, height=np.max(sum_spectrum) * peak_threshold, distance=5)
    peak_frequencies = frequencies[peaks]
    peak_magnitudes = sum_spectrum[peaks]
    
    # Select top N peaks
    top_indices = np.argsort(peak_magnitudes)[-num_top_frequencies:]
    top_frequencies = peak_frequencies[top_indices]
    top_magnitudes = peak_magnitudes[top_indices]
    
    # Sort by frequency
    sorted_indices = np.argsort(top_frequencies)
    top_frequencies = top_frequencies[sorted_indices]
    top_magnitudes = top_magnitudes[sorted_indices]
    
    # Convert to musical notes
    top_notes = [librosa.midi_to_note(librosa.hz_to_midi(f)) for f in top_frequencies]
    
    return list(zip(top_frequencies, top_notes, top_magnitudes))

def print_frequencies(frequencies_data):
    """
    Print extracted frequency data.
    """
    print("Top detected frequencies, notes, and magnitudes:")
    for freq, note, mag in frequencies_data:
        print(f"{freq:.2f} Hz -> {note} (Magnitude: {mag:.2f})")

if __name__ == "__main__":
    file_path = "../dataset/Training/C/C_Electric_Fabi_1.wav"
    frequencies_data = extract_frequencies(file_path)
    print_frequencies(frequencies_data)


'''
in case you want to see the specturm
I kept the code here for reference
'''

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.plot(frequencies, sum_spectrum, color='b', label="Spectrum")

# for freq, note, mag in zip(top_frequencies, top_notes, top_magnitudes):
#     plt.axvline(freq, color='r', linestyle='--', alpha=0.6)
#     plt.text(freq, mag * 1.1, note, color='r', fontsize=12, rotation=45)

# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.title("Summed Spectrogram with Top 6 Notes")
# plt.grid(True)
# plt.xlim(0, sr / 2)
# plt.legend()
# plt.show()

