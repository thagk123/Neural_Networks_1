""" Module providing functions for object serialization and deserialization using pickle """
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


# Συνάρτηση για αποσειριοποίηση
def unpickle(file):
   """ Loads and returns a dictionary from a pickle file. """
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

# Συνάρτηση για τη φόρτωση της CIFAR-10
def load_cifar10_data(folder_path, image):
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
   sc = MinMaxScaler(feature_range=(0, 1))
   train_data = sc.fit_transform(train_data)
   test_data = sc.transform(test_data)
   image = sc.transform(image.view(-1, 3072))

   train_data, train_labels = torch.tensor(train_data).float(), torch.tensor(train_labels)
   test_data, test_labels = torch.tensor(test_data).float(), torch.tensor(test_labels)
   image = torch.tensor(image).float()

   return train_data, train_labels, test_data, test_labels, label_names, image


# Καθορισμός της διαδρομής του φακέλου
folder_path = "C:/Users/gouti/Downloads/cifar-10-python/cifar-10-batches-py"

# Βήμα 1: Φόρτωση εικόνας
image_path = "C:/Users/gouti/PycharmProjects/NN24_1st_Project/ship.jpg"  # Βάλε το σωστό όνομα αρχείου
image = Image.open(image_path).convert("RGB")  # Μετατροπή σε RGB για 3 κανάλια χρώματος

# Βήμα 2: Εφαρμογή μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 έχει διαστάσεις 32x32
    transforms.ToTensor()        # Μετατροπή σε PyTorch Tensor
])

image = transform(image)  # Εφαρμογή των μετασχηματισμών

train_data, train_labels, test_data, test_labels, label_names, image = load_cifar10_data(folder_path, image)

# Δημιουργία DataLoader με batch size
batch_size = 32
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Ορισμός του CNN μοντέλου
class CNN(nn.Module):
    """ Class representing a CNN model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout_1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_2(x)
        x = self.fc2(x)
        return x


model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

# Λίστα για αποθήκευση του loss σε επιλεγμένες εποχές
selected_epochs = []
selected_losses = []

# Early Stopping parameters
patience = 30
best_loss = float('inf')
counter = 0

prev_lr = optimizer.param_groups[0]['lr']

# Λούπα εκπαίδευσης
num_epochs = 10
model.train()
start_time = time.time()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_data, batch_labels in train_loader:
        outputs = model(batch_data.view(len(batch_data), 3, 32, 32))
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # Ελέγχουμε αν το learning rate άλλαξε
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f"Το learning rate άλλαξε από {prev_lr:.6f} σε {current_lr:.6f} στο τέλος της εποχής {epoch}")
        prev_lr = current_lr

    #if (epoch + 1) <= 20 or (epoch + 1) % 50 == 0:
    selected_epochs.append(epoch + 1)
    selected_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    scheduler.step(avg_loss)

    # Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            if (epoch + 1) not in selected_epochs:
                selected_epochs.append(epoch + 1)
                selected_losses.append(avg_loss)
            print(f"Early stopping at epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            break

end_time = time.time()
print(f"Χρόνος εκπαίδευσης: {(end_time - start_time) / 60:.2f} λεπτά")

# Αποθήκευση του Εκπαιδευμένου Classifier:
torch.save(model.state_dict(), "classifier.pth")

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

# Κάνουμε την πρόβλεψη
with torch.no_grad():
    output = model(image.view(1, 3, 32, 32))
    _, predicted = torch.max(output, 1)

# Εκτύπωση του αποτελέσματος
print(f"Η εικόνα ταξινομήθηκε ως: {label_names[predicted]}")
# End-of-file (EOF)
