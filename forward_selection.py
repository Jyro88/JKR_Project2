import random

# Stub evaluation function, returns a random accuracy value
def evaluate_subset(subset):
    return random.uniform(0, 1)

# Greedy search algorithm for forward feature selection
def forward_selection(num_features):   
    # Initialize the best subset with an empty set
    best_subset = set()
    # Print initial accuracy without any features
    best_accuracy = evaluate_subset(best_subset)
    print(f"Using no features and 'random' evaluation, I get an accuracy of {best_accuracy * 100:.1f}% \n")
    subset_evaluations = {}
    overall_best_accuracy = best_accuracy
    
    # Iterate over the number of features
    for i in range(1, num_features + 1):
        current_best_feature = None
        improved = False
        
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
                    improved = True
        
        # If no improvement, terminate the search
        if not improved:
            print("No improvement from adding any features, terminating search.")
            break
        
        # Add the best feature to the best subset
        best_subset.add(current_best_feature)
        overall_best_accuracy = best_accuracy
        print(f"\n Feature set {best_subset} was best, accuracy is {best_accuracy * 100:.1f}% \n")
    
    return best_subset, overall_best_accuracy * 100