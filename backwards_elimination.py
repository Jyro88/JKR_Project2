from forward_selection import evaluate_subset

# Greedy search algorithm for backward feature elimination
def backward_elimination(num_features):   
    # Initialize the best subset with all features
    best_subset = set(range(1, num_features + 1))
    best_accuracy = evaluate_subset(best_subset)
    subset_evaluations = {}
    
    # Iterate while there are features in the best subset
    while len(best_subset) > 1:
        current_worst_feature = None
        
        # Iterate over all features in the best subset
        for feature in list(best_subset):
            # Remove the feature from the current subset
            current_subset = best_subset.copy()
            current_subset.remove(feature)
            
            # Evaluate the current subset
            accuracy = evaluate_subset(current_subset)
            subset_evaluations[str(current_subset)] = accuracy
            
            # Print the evaluation of the current subset
            print(f"Using feature(s) {current_subset} accuracy is {accuracy * 100:.1f}%")
            
            # If the accuracy is higher than the best accuracy, update the worst feature
            if accuracy > best_accuracy:
                current_worst_feature = feature
                best_accuracy = accuracy
        
        # Remove the worst feature from the best subset
        if current_worst_feature is not None:
            best_subset.remove(current_worst_feature)
            print(f"\n Feature set {best_subset} was best, accuracy is {best_accuracy * 100:.1f}% \n")
        else:
            # If no improvement, terminate the search
            print("No improvement, terminating search.")
            break
    
    return best_subset, best_accuracy * 100