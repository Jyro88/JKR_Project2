import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load datasets
small_dataset = np.loadtxt('small-test-dataset.txt')
large_dataset = np.loadtxt('large-test-dataset.txt')

# Normalize datasets
def normalize_data(dataset):
    features = dataset[:, 1:]
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    normalized_dataset = np.column_stack((dataset[:, 0], normalized_features))
    return normalized_dataset

normalized_small_dataset = normalize_data(small_dataset)
normalized_large_dataset = normalize_data(large_dataset)

# Function to plot dataset
def plot_dataset(dataset, feature_indices, title):
    classes = dataset[:, 0]
    features = dataset[:, 1:]
    
    feature_x = features[:, feature_indices[0]]
    feature_y = features[:, feature_indices[1]]
    
    plt.scatter(feature_x[classes == 1], feature_y[classes == 1], color='red', label='Class 1')
    plt.scatter(feature_x[classes == 2], feature_y[classes == 2], color='blue', label='Class 2')
    plt.xlabel(f'Feature {feature_indices[0] + 1}')
    plt.ylabel(f'Feature {feature_indices[1] + 1}')
    plt.title(title)
    plt.legend()

# Plot original and normalized datasets
plt.figure(figsize=(14, 10))

# Small dataset, original features
plt.subplot(2, 2, 1)
plot_dataset(small_dataset, [2, 4], 'Small Dataset (Original Features 3 & 5)')

# Small dataset, normalized features
plt.subplot(2, 2, 2)
plot_dataset(normalized_small_dataset, [2, 4], 'Small Dataset (Normalized Features 3 & 5)')

# Large dataset, original features
plt.subplot(2, 2, 3)
plot_dataset(large_dataset, [29, 31], 'Large Dataset (Original Features 30 & 32)')

# Large dataset, normalized features
plt.subplot(2, 2, 4)
plot_dataset(normalized_large_dataset, [29, 31], 'Large Dataset (Normalized Features 30 & 32)')

plt.tight_layout()
plt.show()
