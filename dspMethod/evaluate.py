import numpy as np

def identify_chord(frequencies_data):
    """
    Identify the most likely chord based on detected frequencies.
    """
    chords = {
        "Am": [110.00, 164.81, 220.00],
        "Bb": [116.54, 174.61, 233.08],
        "Bdim": [123.47, 174.61, 246.94],
        "C": [130.81, 196.00, 261.63],
        "Dm": [146.83, 220.00, 293.66],
        "Em": [164.81, 246.94, 329.63],
        "F": [174.61, 261.63, 349.23],
        "G": [196.00, 293.66, 392.00]
    }
    
    chord_frequencies_map = {}
    for chord, base_freqs in chords.items():
        chord_set = set()
        for n in range(-1, 6):
            chord_set.update(np.array(base_freqs) * (2 ** n))
        chord_frequencies_map[chord] = np.array(sorted(chord_set))
    
    # Normalize detected frequencies
    def normalize_to_octave(frequencies, base_octave=130.81):
        return base_octave * (2 ** ((np.log2(frequencies / base_octave)) % 1))
    
    normalized_input_frequencies = normalize_to_octave(np.array([f[0] for f in frequencies_data]))
    input_magnitudes = np.array([f[2] for f in frequencies_data])
    
    normalized_chord_frequencies_map = {
        chord: normalize_to_octave(chord_frequencies)
        for chord, chord_frequencies in chord_frequencies_map.items()
    }
    
    # Compute weighted differences
    chord_scores = {}
    for chord, chord_frequencies in normalized_chord_frequencies_map.items():
        weighted_sum = 0
        for freq, mag in zip(normalized_input_frequencies, input_magnitudes):
            closest_freq = chord_frequencies[np.argmin(np.abs(chord_frequencies - freq))]
            diff = np.abs(freq - closest_freq)
            weighted_sum += diff * mag
        chord_scores[chord] = weighted_sum
    
    best_match = min(chord_scores, key=chord_scores.get)
    return best_match