import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

data = np.load('chord_test_dataset.npy', allow_pickle=True)
X = np.array([sample[0] for sample in data], dtype=np.float32)
y = np.array([sample[1] for sample in data])

scaler = MinMaxScaler()
X_reshaped = X.reshape(-1, 12)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 6, 2)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

class ChordClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChordClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = 12
hidden_dim = 32
output_dim = num_classes
model = ChordClassifier(input_dim, hidden_dim, output_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load("chord_classifier.pth"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    with tqdm(test_loader, desc="Testing on Test Dataset") as pbar:
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = outputs.argmax(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            pbar.set_postfix(accuracy=correct / total)

test_acc = correct / total
print(f"\n✅ Test Accuracy on Test Dataset: {test_acc:.4f}")

def predict_chord(sample):
    sample = torch.tensor(scaler.transform(sample.reshape(1, -1)), dtype=torch.float32).reshape(1, 6, 2).to(device)
    with torch.no_grad():
        output = model(sample)
        predicted_label = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([predicted_label])[0]

print("Prediction Example from Test Dataset:", predict_chord(X_tensor[0]))
