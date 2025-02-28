import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


file_path = "../dataset/Training/C/C_Electric_Fabi_2.wav"
y, sr = librosa.load(file_path, sr=None)


D = librosa.stft(y, n_fft=2048, hop_length=512)
S = np.abs(D)


sum_spectrum = np.sum(S, axis=1)

frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)

peaks, properties = find_peaks(sum_spectrum, height=np.max(sum_spectrum) * 0.05, distance=5)

peak_frequencies = frequencies[peaks]
peak_magnitudes = sum_spectrum[peaks]

top_indices = np.argsort(peak_magnitudes)[-6:]
top_frequencies = peak_frequencies[top_indices]
top_magnitudes = peak_magnitudes[top_indices]

sorted_indices = np.argsort(top_frequencies)
top_frequencies = top_frequencies[sorted_indices]
top_magnitudes = top_magnitudes[sorted_indices]


top_notes = [librosa.midi_to_note(librosa.hz_to_midi(f)) for f in top_frequencies]

plt.figure(figsize=(10, 5))
plt.plot(frequencies, sum_spectrum, color='b', label="Spectrum")

for freq, note, mag in zip(top_frequencies, top_notes, top_magnitudes):
    plt.axvline(freq, color='r', linestyle='--', alpha=0.6)
    plt.text(freq, mag * 1.1, note, color='r', fontsize=12, rotation=45)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Summed Spectrogram with Top 6 Notes")
plt.grid(True)
plt.xlim(0, sr / 2)
plt.legend()
plt.show()

print("Top 6 detected frequencies and notes:")
for freq, note in zip(top_frequencies, top_notes):
    print(f"{freq:.2f} Hz -> {note}")
