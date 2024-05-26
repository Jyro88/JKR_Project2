import numpy as np
from sklearn.preprocessing import StandardScaler
from time import time

class NearestNeighborClassifier:
    def __init__(self):
        self.training_data = None
        self.training_labels = None
        self.scaler = StandardScaler()
        self.trace = []

    def train(self, data, labels):
        start_time = time()
        self.training_data = self.scaler.fit_transform(data)
        self.training_labels = labels
        end_time = time()
        self.trace.append(f"Training completed in {end_time - start_time:.4f} seconds.")

    def predict(self, instance):
        start_time = time()
        instance = self.scaler.transform([instance])[0]
        distances = np.linalg.norm(self.training_data - instance, axis=1)
        nearest_index = np.argmin(distances)
        prediction = self.training_labels[nearest_index]
        end_time = time()
        self.trace.append(f"Prediction for instance completed in {end_time - start_time:.4f} seconds.")
        return prediction

    def get_trace(self):
        return self.trace

class LeaveOneOutValidator:
    def __init__(self, classifier):
        self.classifier = classifier
        self.trace = []

    def validate(self, data, labels, feature_subset):
        start_time = time()
        data_subset = data[:, feature_subset]
        correct_predictions = 0

        for i in range(len(data_subset)):
            train_data = np.delete(data_subset, i, axis=0)
            train_labels = np.delete(labels, i)
            test_instance = data_subset[i]
            true_label = labels[i]

            self.classifier.train(train_data, train_labels)
            prediction = self.classifier.predict(test_instance)

            if prediction == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(data_subset)
        end_time = time()
        self.trace.append(f"Validation completed in {end_time - start_time:.4f} seconds with accuracy: {accuracy:.4f}.")
        return accuracy

    def get_trace(self):
        return self.trace

def load_data(filename):
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels

# Load the dataset
features, labels = load_data('small-test-dataset-1.txt')
# features, labels = load_data('large-test-dataset-1.txt')

# Normalize the features
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Use only features {3, 5, 7} (index 2, 4, 6 in zero-based indexing)
feature_subset = [2, 4, 6]
# feature_subset = [0, 14, 26]

# Initialize the classifier and validator
nn_classifier = NearestNeighborClassifier()
validator = LeaveOneOutValidator(nn_classifier)

# Perform validation
accuracy = validator.validate(features, labels, feature_subset)

# Print the trace logs
print("Classifier Trace Logs:")
for log in nn_classifier.get_trace():
    print(log)

print("\nValidator Trace Logs:")
for log in validator.get_trace():
    print(log)

print(f"\nAccuracy: {accuracy:.4f}")
