import os
from freqSpectrum import extract_frequencies
from evaluate import identify_chord

dataset_path = "../dataset/Training/C/"

total_files = 0
correct_predictions = 0

for filename in os.listdir(dataset_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(dataset_path, filename)
        print(f"Processing: {file_path}")
        

        frequencies_data = extract_frequencies(file_path)
        

        predicted_chord = identify_chord(frequencies_data)
        

        if predicted_chord == "C":
            correct_predictions += 1
        
        total_files += 1


accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0

print(f"\nTotal WAV files processed: {total_files}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
