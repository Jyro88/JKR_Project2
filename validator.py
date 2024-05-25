import numpy as np
from nearest_neighbor_classifier import train
from nearest_neighbor_classifier import test

class Validator:
    def __init__(self, classifier, data, labels):
        self.classifier = classifier
        self.data = data
        self.labels = labels
    
    def leave_one_out_validation(self, feature_subset):
        correct_predictions = 0
        n = len(self.data)
        
        for i in range(n):
            # Prepare training data excluding the i-th instance
            train_data = np.delete(self.data, i, axis=0)
            train_labels = np.delete(self.labels, i, axis=0)
            
            # Train the classifier on the remaining data
            self.classifier.train(train_data[:, feature_subset], train_labels)
            
            # Test the classifier on the left-out instance
            test_instance = self.data[i, feature_subset]
            predicted_label = self.classifier.test(test_instance)
            
            # Check if the prediction is correct
            if predicted_label == self.labels[i]:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / n
        return accuracy
