import random

# Stub evaluation function, returns a random accuracy value
def evaluate_subset(subset):
    return random.uniform(0, 1)

# Greedy search algorithm for forward feature selection
def forward_selection(num_features):   
    # Initialize the best subset with an empty set
    best_subset = set()
    best_accuracy = 0
    subset_evaluations = {}
    
    # Iterate over the number of features
    for i in range(1, num_features + 1):
        current_best_feature = None
        # Iterate over all features
        for feature in range(1, num_features + 1):
            if feature not in best_subset:
                # Add the feature to the current subset
                current_subset = best_subset.copy()
                current_subset.add(feature)
                
                # Evaluate the current subset
                accuracy = evaluate_subset(current_subset)
                subset_evaluations[str(current_subset)] = accuracy
                
                # Print the evaluation of the current subset
                print(f"Using feature(s) {current_subset} accuracy is {accuracy * 100:.1f}%")
                
                # If the accuracy is higher than the best accuracy, update the best feature
                if accuracy > best_accuracy:
                    current_best_feature = feature
                    best_accuracy = accuracy
        
        # Add the best feature to the best subset
        if current_best_feature is not None:
            best_subset.add(current_best_feature)
            print(f"\n Feature set {best_subset} was best, accuracy is {best_accuracy * 100:.1f}% \n")
        else:
            # If no improvement, terminate the search
            print("No improvement, terminating search.")
            break
    
    return best_subset, best_accuracy * 100