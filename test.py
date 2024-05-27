import numpy as np
from nearest_neighbor_classifier import Classifier
from validator import Validator
from data_loader import load_data

def test_accuracy(data, labels, feature_subset):
    classifier = Classifier()
    validator = Validator(classifier, data, labels)
    accuracy = validator.evaluate(feature_subset)
    return accuracy

def main():
    # Test on Small Dataset
    small_data_file = 'small-test-dataset.txt'
    small_data, small_labels = load_data(small_data_file)
    small_features = [2, 4, 6]  # Adjusted for 0-based indexing
    small_accuracy = test_accuracy(small_data, small_labels, small_features)
    print(f"Small Dataset: Using features {small_features}, accuracy is {small_accuracy * 100:.2f}%")
    
    # Test on Large Dataset
    large_data_file = 'large-test-dataset.txt'
    large_data, large_labels = load_data(large_data_file)
    large_features = [0, 14, 26]  # Adjusted for 0-based indexing
    large_accuracy = test_accuracy(large_data, large_labels, large_features)
    print(f"Large Dataset: Using features {large_features}, accuracy is {large_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
