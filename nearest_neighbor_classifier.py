import numpy as np

class NearestNeighborClassifier:
    def __init__(self):
        self.training_data = None
        self.training_labels = None
    
    def train(self, data, labels):
        self.training_data = data
        self.training_labels = labels
    
    def test(self, instance):
        if self.training_data is None or self.training_labels is None:
            raise ValueError("Classifier has not been trained yet.")
        
        # Compute Euclidean distances from the test instance to all training instances
        distances = np.linalg.norm(self.training_data - instance, axis=1)
        
        # Find the index of the nearest neighbor
        nearest_neighbor_idx = np.argmin(distances)
        
        # Return the label of the nearest neighbor
        return self.training_labels[nearest_neighbor_idx]
