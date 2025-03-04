import os
import numpy as np
import librosa
from freqSpectrum import extract_frequencies

def process_dataset(dataset_root, output_filename):
    output_data = []

    for chord_label in os.listdir(dataset_root):
        chord_path = os.path.join(dataset_root, chord_label)

        if not os.path.isdir(chord_path):
            continue

        print(f"Processing chord: {chord_label}")

        for filename in os.listdir(chord_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(chord_path, filename)
                print(f"  - Processing file: {file_path}")

                frequencies_data = extract_frequencies(file_path)

                if len(frequencies_data) < 6:
                    print(f"  ⚠ Warning: {file_path} has less than 6 frequency components. Padding with (0, 0)...")
                    while len(frequencies_data) < 6:
                        frequencies_data.append((0.0, "N/A", 0.0))  # 使用 (频率=0, 无效音符, 幅值=0) 进行填充

                frequencies_data = frequencies_data[:6]
                formatted_data = np.array([[freq, mag] for freq, _, mag in frequencies_data])
                output_data.append((formatted_data, chord_label))

    output_array = np.array(output_data, dtype=object)
    np.save(output_filename, output_array)
    print(f"\n ✅ Dataset saved as '{output_filename}'. Total samples: {len(output_data)}")


# 处理训练集和测试集
process_dataset("../dataset/Training/", "chord_train_dataset.npy")
process_dataset("../dataset/Test/", "chord_test_dataset.npy")
