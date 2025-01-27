import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Συνάρτηση για αποσειριοποίηση
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Συνάρτηση για τη φόρτωση της CIFAR-10
def load_cifar10_data(folder_path):
    train_data = None
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(f"{folder_path}/data_batch_{i}")
        batch_data = batch[b'data']
        if train_data is None:
            train_data = batch_data
        else:
            train_data = np.concatenate((train_data, batch_data), axis=0)
        train_labels.extend(batch[b'labels'])

    test_batch = unpickle(f"{folder_path}/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Φόρτωση των metadata για τα labels
    meta_data = unpickle(f"{folder_path}/batches.meta")
    label_names = meta_data[b'label_names']
    label_names = [label.decode('utf-8') for label in label_names]  # Μετατροπή των ετικετών από bytes σε string

    # Τυποποίηση δεδομένων
    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    train_data, train_labels = torch.tensor(train_data).float(), torch.tensor(train_labels)
    test_data, test_labels = torch.tensor(test_data).float(), torch.tensor(test_labels)

    return train_data, train_labels, test_data, test_labels, label_names


# Καθορισμός της διαδρομής του φακέλου
folder_path = "C:/Users/gouti/Downloads/cifar-10-python/cifar-10-batches-py"
train_data, train_labels, test_data, test_labels, label_names = load_cifar10_data(folder_path)

# Δημιουργία DataLoader με batch size
batch_size = 1024
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Ορισμός του MLP μοντέλου
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Το τελικό επίπεδο χωρίς ReLU
        return x


model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# Λίστα για αποθήκευση του loss σε επιλεγμένες εποχές
selected_epochs = []
selected_losses = []

# Λούπα εκπαίδευσης
num_epochs = 100
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_data, batch_labels in train_loader:
        outputs = model(batch_data.view(len(batch_data), 3, 32, 32))
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    if (epoch + 1) <= 20 or (epoch + 1) % 50 == 0:
        selected_epochs.append(epoch + 1)
        selected_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

end_time = time.time()
print(f"Χρόνος εκπαίδευσης: {(end_time - start_time) / 60:.2f} λεπτά")

# Δημιουργία διαγράμματος για το loss
plt.figure(figsize=(10, 6))
plt.plot(selected_epochs, selected_losses, label="Selected Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss σε Επιλεγμένες Εποχές")
plt.legend()
plt.grid()
plt.show()

# Αξιολόγηση του μοντέλου
model.eval()

def predict_acc(data, labels, name):
    with torch.no_grad():
        outputs = model(data.view(len(data), 3, 32, 32))
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)

    print(f'Ακρίβεια στο {name} set: {accuracy:.2f}%')

    return predicted

predict_acc(train_data, train_labels, "Training")
predicted = predict_acc(test_data, test_labels, "Test")

# Υπολογισμός ακρίβειας ανά κατηγορία
def accuracy_per_category(test_labels, predicted, label_names):
    class_correct = []
    class_total = []

    for i in range(10):
        class_correct.append(0)
        class_total.append(0)

    for i in range(10000):
        label = test_labels[i].item()
        class_total[label] += 1
        if predicted[i].item() == label:
            class_correct[label] += 1

    for i in range(10):
        if class_total[i] > 0:
          accuracy = 100 * class_correct[i] / class_total[i]
        else:
            accuracy = 0
        print(f"Κατηγορία: {label_names[i]:<10s} | Σωστά: {class_correct[i]:<3} / {class_total[i]:<3} | Ακρίβεια: {accuracy:.2f}%")

accuracy_per_category(test_labels, predicted, label_names)

# Εντοπισμός σωστών και λανθασμένων κατηγοριοποιήσεων
correct_examples = []
incorrect_examples = []

for i in range(10000):
    true_label = label_names[test_labels[i].item()]
    predicted_label = label_names[predicted[i].item()]
    if predicted[i].item() == test_labels[i].item():
        correct_examples.append((i, true_label, predicted_label))
    else:
        incorrect_examples.append((i, true_label, predicted_label))

def print_examples(examples, label):
    print(f"\nΧαρακτηριστικά {label} Κατηγοριοποιημένα Δείγματα:")
    for idx, true_label, predicted_label in examples[:10]:  # Εμφάνιση των 10 πρώτων
        print(f"Δείγμα {idx}: Πραγματική Ετικέτα = {true_label}, Προβλεπόμενη Ετικέτα = {predicted_label}")

print_examples(correct_examples, "Σωστά")
print_examples(incorrect_examples, "Εσφαλμένα")