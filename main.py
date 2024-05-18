from forward_selection import forward_selection
from forward_selection import evaluate_subset

# Main function
def main():
    print("Welcome to JKR's Feature Selection Algorithm.")
    num_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algorithm_choice = int(input())
    
    # Print initial accuracy without any features
    no_feature_accuracy = evaluate_subset(set())
    print(f"Using no features and 'random' evaluation, I get an accuracy of {no_feature_accuracy * 100:.1f}%")
    
    if algorithm_choice == 1:
        print("Beginning search.")
        # Perform forward selection
        best_subset, best_accuracy = forward_selection(num_features)
        # Print results
        print("\nFinished search!! The best feature subset is", best_subset, "which has an accuracy of", best_accuracy, "%")
    else:
        print("You selected an invalid option.")

if __name__ == "__main__":
    main()
