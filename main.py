import random

# Stub evaluation function, returns a random accuracy value
def evaluate_subset(subset):
    return random.uniform(0, 1)

# Generate all possible subsets
# Uses bit manipulation which makes it the most optimal way of getting all of the subsets I think
def generate_subsets(num_features):
    # Initialize an empty list to store subsets
    subsets = []

    # Iterate through all numbers from 1 to 2^num_features - 1
    # (excluding 0 to avoid the empty subset)
    for i in range(1, 2**num_features):

        # Initialize an empty set to represent the current subset
        subset = set()

        # This checks if the current feature number index is a 1 in the binary representation of i
        # i goes from 1 to 2^num features
        # Ex: When we get to the fifth subset, 5 = 101
        # The 1st and 3rd index is 1, so this would add the subset {1, 3}
        for j in range(num_features):
            if i & (1 << j):
                subset.add(j + 1)  

        subsets.append(subset)

    return subsets


# Greedy search algorithm
def greedy_search(num_features):
    # Generate all possible subsets
    subsets = generate_subsets(num_features)
    
    # Initialize best subset and accuracy
    best_subset = set()
    best_accuracy = 0
    all_results = {}
    
    # Iterate through all possible feature subsets
    for subset in subsets:
        accuracy = evaluate_subset(subset)
        all_results[str(subset)] = accuracy
        
        # Update the best subset if the accuracy is higher
        if accuracy > best_accuracy:
            best_subset = subset
            best_accuracy = accuracy
    
    return best_subset, best_accuracy, all_results

# Main function
def main():
    # Take number of features as input from the user
    num_features = int(input("Enter the total number of features: "))
    
    # Perform greedy search
    best_subset, best_accuracy, all_results = greedy_search(num_features)
    
    # Print all results
    print("All subset evaluations:")
    for subset, accuracy in all_results.items():
        print(f"Subset: {subset}, Accuracy: {accuracy}")
    
    # Print results
    print("\nBest subset:", best_subset)
    print("Best accuracy:", best_accuracy)

if __name__ == "__main__":
    main()
