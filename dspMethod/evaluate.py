import numpy as np

chords = {
    "Am": [110.00, 164.81, 220.00],  # A, C, E
    "Bb": [116.54, 174.61, 233.08], # Bb, D, F
    "Bdim": [123.47, 174.61, 246.94], # B, D, F
    "C": [130.81, 196.00, 261.63], # C, E, G
    "Dm": [146.83, 220.00, 293.66], # D, F, A
    "Em": [164.81, 246.94, 329.63], # E, G, B
    "F": [174.61, 261.63, 349.23], # F, A, C
    "G": [196.00, 293.66, 392.00] # G, B, D
}

chord_frequencies_map = {}
for chord, base_freqs in chords.items():
    chord_set = set()
    for n in range(-1, 6):
        chord_set.update(np.array(base_freqs) * (2 ** n))
    chord_frequencies_map[chord] = np.array(sorted(chord_set))


input_frequencies = np.array([132.81, 195.31, 265.62, 992.19, 1367.19, 1570.31])
input_magnitudes = np.array([1257.33, 862.88, 1055.20, 706.77, 1290.29, 487.30])

# octave normalization
def normalize_to_octave(frequencies, base_octave=130.81, target_octave=261.63):
    return base_octave * (2 ** ((np.log2(frequencies / base_octave)) % 1))


normalized_input_frequencies = normalize_to_octave(input_frequencies)

normalized_chord_frequencies_map = {
    chord: normalize_to_octave(chord_frequencies)
    for chord, chord_frequencies in chord_frequencies_map.items()
}

chord_scores = {}
for chord, chord_frequencies in normalized_chord_frequencies_map.items():
    weighted_sum = 0
    for freq, mag in zip(normalized_input_frequencies, input_magnitudes):
        closest_freq = chord_frequencies[np.argmin(np.abs(chord_frequencies - freq))]
        diff = np.abs(freq - closest_freq)
        weighted_sum += diff * mag
    chord_scores[chord] = weighted_sum
    print(f"{chord}: Total Weighted Difference = {weighted_sum:.2f}")

best_match = min(chord_scores, key=chord_scores.get)
print(f"Most Likely Chord: {best_match}")
