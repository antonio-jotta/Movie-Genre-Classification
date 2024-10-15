from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


def train_model(X_train, y_train, X_val, y_val):
    """
    Trains a k-NN model

    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        X_val: Feature matrix for validation.
        y_val: Labels for validation.

    Returns:
        best_model: The best trained k-NN model.
        k_values: Range of k values evaluated.
        accuracy_scores: List of accuracy scores for each k.
    """
    # Get the unique genres in the order that corresponds to the model's internal representation
    unique_genres = np.unique(y_train)

    best_k = None
    best_accuracy = 0  # To track the best accuracy score
    best_model = None
    accuracy_scores = []  # List to store accuracy scores for each k

    k_values = range(1, 101)  # Loop through k values from 1 to 100
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)

        # Train the model
        model.fit(X=X_train, y=y_train)

        # Get predicted probabilities
        predictions_proba = model.predict_proba(X=X_val)

        # Convert probabilities to predicted class indices using argmax
        predicted_class_indices = np.argmax(predictions_proba, axis=1)

        # Map predicted indices back to genre labels using unique_genres
        predictions = [unique_genres[idx] for idx in predicted_class_indices]

        # Calculate the accuracy between true labels and predicted labels
        current_accuracy = accuracy_score(y_true=y_val, y_pred=predictions)
        accuracy_scores.append(current_accuracy)  # Save the accuracy score

        # If this accuracy is better than the best one, update best_k and best_accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = k
            best_model = model

        # print(f"Tested k={k}: Accuracy = {current_accuracy}")

    print(f"\nBest k={best_k} with Accuracy = {best_accuracy}")

    # Now calculate the F1 score for the best model based on accuracy
    predictions_proba_best = best_model.predict_proba(X=X_val)
    predicted_class_indices_best = np.argmax(predictions_proba_best, axis=1)
    predictions_best = [unique_genres[idx] for idx in predicted_class_indices_best]

    # Calculate and print the F1 score for the best model
    best_f1 = f1_score(y_true=y_val, y_pred=predictions_best, average='weighted')
    print(f"F1 Score for best model (k={best_k}): {best_f1}")

    return best_model, k_values, accuracy_scores


def plot_acc_vs_k(k_values, acc_scores):
    """
    Plots the accuracy scores as a function of k values.

    Args:
        k_values (list or array-like): The list of k values.
        acc_scores (list or array-like): The list of accuracy scores corresponding to the k values.

    Returns:
        None: Saves the plot.
    """
    plt.figure()
    plt.plot(k_values, acc_scores)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../plots/knn_acc")


def test_model(model, test):
    """
    Tests the trained k-NN model on a new dataset with both plot and director features.

    Args:
        model: The trained k-NN model.
        test (pd.DataFrame): The test dataset with 'Plot' and 'Director' columns.

    Returns:
        np.ndarray: Predicted probabilities for each genre.
    """
    return model.predict_proba(test)