import numpy as np
from validator import evaluate_subset

def forward_selection(data, labels):
    num_features = data.shape[1]
    best_subset = set()
    best_accuracy = evaluate_subset(best_subset, data, labels)
    print(f"Using no features, I get an accuracy of {best_accuracy * 100:.1f}% \n")
    
    subset_evaluations = {}
    overall_best_accuracy = best_accuracy

    for i in range(num_features):
        current_best_feature = None
        improved = False

        for feature in range(num_features):
            if feature not in best_subset:
                current_subset = best_subset.copy()
                current_subset.add(feature)

                accuracy = evaluate_subset(current_subset, data, labels)
                subset_evaluations[str(current_subset)] = accuracy

                print(f"Using feature(s) {current_subset} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_accuracy:
                    current_best_feature = feature
                    best_accuracy = accuracy
                    improved = True

        if not improved:
            print("No improvement from adding any features, terminating search.")
            break

        if current_best_feature is not None:
            best_subset.add(current_best_feature)
            overall_best_accuracy = best_accuracy
            print(f"\nFeature set {best_subset} was best, accuracy is {best_accuracy * 100:.1f}% \n")

    return best_subset, overall_best_accuracy * 100
