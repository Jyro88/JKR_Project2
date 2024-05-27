from validator import evaluate_subset

def backward_elimination(data, labels):
    num_features = data.shape[1]
    best_subset = set(range(num_features))
    best_accuracy = evaluate_subset(best_subset, data, labels)
    print(f"Using feature(s) {best_subset} accuracy is {best_accuracy * 100:.1f}%")
    overall_best_accuracy = best_accuracy

    while len(best_subset) > 1:
        current_worst_feature = None

        for feature in list(best_subset):
            current_subset = best_subset.copy()
            current_subset.remove(feature)
            accuracy = evaluate_subset(current_subset, data, labels)

            print(f"Using feature(s) {current_subset} accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_accuracy:
                current_worst_feature = feature
                best_accuracy = accuracy

        if current_worst_feature is None:
            print("No improvement from removing any features, terminating search.")
            break

        best_subset.remove(current_worst_feature)
        overall_best_accuracy = best_accuracy
        print(f"\n Feature set {best_subset} was best, accuracy is {best_accuracy * 100:.1f}% \n")

    return best_subset, overall_best_accuracy * 100
