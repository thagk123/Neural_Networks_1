import pickle
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

start_time = time.time()

# Συνάρτηση για αποσειριοποίηση
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Συνάρτηση για τη φόρτωση της CIFAR-10
def load_cifar10_data(folder_path):
    # Αρχικοποίηση του τελικού πίνακα δεδομένων και των labels
    train_data = None
    train_labels = []
    for i in range(1, 6):
       batch = unpickle(f"{folder_path}/data_batch_{i}")
       batch_data=batch[b'data']
       if train_data is None:
           train_data = batch_data
       else:
           train_data = np.concatenate((train_data, batch_data), axis=0) #Συγχώνευση του επόμενου batch στον τελικό πίνακα
       batch_labels=batch[b'labels']
       for k in batch_labels:
           train_labels.append(k) #Προσθήκη των labels του επόμενου batch στον τελική λίστα

    # Φόρτωση των δεδομένων ελέγχου
    test_batch = unpickle(f"{folder_path}/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Φόρτωση των metadata για τα labels
    meta_data = unpickle(f"{folder_path}/batches.meta")
    label_names = meta_data[b'label_names']
    label_names = [label.decode('utf-8') for label in label_names] # Μετατροπή των ετικετών από bytes σε string

    # Μετατροπή σε μορφή PyTorch tensors
    #train_data, train_labels = torch.tensor(train_data).float(), torch.tensor(train_labels)
    #test_data, test_labels = torch.tensor(test_data).float(), torch.tensor(test_labels)

    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    pca = PCA(n_components=0.95)
    train_data = pca.fit_transform(train_data)
    test_data = pca.transform(test_data)

    print("Αριθμός συνιστωσών που διατηρήθηκαν:", pca.n_components_)

    return train_data, train_labels, test_data, test_labels


# Καθορισμός της διαδρομής του φακέλου
folder_path = "C:/Users/gouti/Downloads/cifar-10-python/cifar-10-batches-py"
train_data, train_labels, test_data, test_labels = load_cifar10_data(folder_path)

end_time = time.time()
print(f"Χρόνος εκτέλεσης για φορτώση των δεδομένων CIFAR-10 και εφαρμογή preprocessing (scaling, PCA) : {end_time - start_time:.2f} δευτερόλεπτα")

# Υλοποίηση k-NN classifier
def knn_predict(train_data, train_labels, test_images, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    start_time = time.time()
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_images)
    end_time = time.time()
    print(f"Χρόνος εκτέλεσης για k-NN με k={k}: {end_time - start_time:.2f} δευτερόλεπτα")

    return predictions

# Υλοποίηση NCC classifier
def nearest_centroid_predict(train_data, train_labels, test_images):
    classifier = NearestCentroid()
    start_time = time.time()
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_images)
    end_time = time.time()
    print(f"Χρόνος εκτέλεσης για Nearest Centroid Classifier: {end_time - start_time:.2f} δευτερόλεπτα")

    return predictions

# Ακρίβεια μοντέλων
def accuracy(predictions, labels):
    return accuracy_score(predictions, labels)


# Υπολογισμός ακρίβειας για k-NN με k=1 και k=3 στο test set
knn_predictions_k1 = knn_predict(train_data, train_labels, test_data, k=1)
knn_predictions_k3 = knn_predict(train_data, train_labels, test_data, k=3)
accuracy_knn_k1 = accuracy(knn_predictions_k1, test_labels)
accuracy_knn_k3 = accuracy(knn_predictions_k3, test_labels)

# Υπολογισμός ακρίβειας για k-NN με k=1 και k=3 στο training set
knn_train_predictions_k1 = knn_predict(train_data, train_labels, train_data, k=1)
knn_train_predictions_k3 = knn_predict(train_data, train_labels, train_data, k=3)
accuracy_knn_train_k1 = accuracy(knn_train_predictions_k1, train_labels)
accuracy_knn_train_k3 = accuracy(knn_train_predictions_k3, train_labels)

# Υπολογισμός ακρίβειας για τον NCC classifier στο test set
nearest_centroid_predictions = nearest_centroid_predict(train_data, train_labels, test_data)
accuracy_nearest_centroid = accuracy(nearest_centroid_predictions, test_labels)

# Υπολογισμός ακρίβειας για τον NCC classifier στο training set
nearest_centroid_train_predictions = nearest_centroid_predict(train_data, train_labels, train_data)
accuracy_nearest_centroid_train = accuracy(nearest_centroid_train_predictions, train_labels)

print(f"Ακρίβεια k-NN με k=1 στο test set: {accuracy_knn_k1*100:.2f}%")
print(f"Ακρίβεια k-NN με k=3 στο test set: {accuracy_knn_k3*100:.2f}%")
print(f"Ακρίβεια k-NN με k=1 στο training set: {accuracy_knn_train_k1*100:.2f}%")
print(f"Ακρίβεια k-NN με k=3 στο training set: {accuracy_knn_train_k3*100:.2f}%")
print(f"Ακρίβεια NCC στο test set: {accuracy_nearest_centroid*100:.2f}%")
print(f"Ακρίβεια NCC στο training set: {accuracy_nearest_centroid_train*100:.2f}%")
