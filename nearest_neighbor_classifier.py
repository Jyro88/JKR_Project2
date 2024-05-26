import numpy as np

class NNClassifier:
    def __init__(self):
        self.training_data = None
        self.labels = None

    def train(self, data, labels):
        self.training_data = data
        self.labels = labels

    def test(self, instance):
        distances = np.linalg.norm(self.training_data - instance, axis=1)
        nearest_index = np.argmin(distances)
        return self.labels[nearest_index]

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out_validation(self, data, labels, feature_subset):
        n_instances = data.shape[0]
        correct_predictions = 0

        for i in range(n_instances):
            train_data = np.delete(data, i, axis=0)
            train_labels = np.delete(labels, i)
            test_instance = data[i, :]
            true_label = labels[i]

            self.classifier.train(train_data[:, feature_subset], train_labels)
            predicted_label = self.classifier.test(test_instance[feature_subset])

            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / n_instances
        return accuracy

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

# Instantiate the classifier and validator
classifier = NNClassifier()
validator = Validator(classifier)

# Calculate the accuracy using the leave-one-out validation method
accuracy = validator.leave_one_out_validation(features, labels, feature_subset)
print(f'Accuracy: {accuracy:.2f}')
