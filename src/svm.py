from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt


def train_model(train, validation):
    """
    Trains an SVM model using TfidfVectorizer for plots and OneHotEncoder for directors.
    It tunes the C parameter and tests multiple kernels, plotting the F1 score against different C values for the best kernel.

    Args:
        train (pd.DataFrame): Training dataset with 'Plot', 'Director', and 'Genre' columns.
        validation (pd.DataFrame): Validation dataset with 'Plot', 'Director', and 'Genre' columns.

    Returns:
        best_model: The trained SVM model.
        vectorizer: The fitted TfidfVectorizer used for text transformation.
        encoder: The fitted OneHotEncoder used for encoding directors.
    """
    # Vectorize the movie plots
    vectorizer = TfidfVectorizer()
    X_train_plot = vectorizer.fit_transform(train['Plot'])
    X_val_plot = vectorizer.transform(validation['Plot'])

    # One-hot encode the directors
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_director = encoder.fit_transform(train[['Director']])
    X_val_director = encoder.transform(validation[['Director']])

    # Combine plot and director features
    X_train = hstack([X_train_plot, X_train_director])
    X_val = hstack([X_val_plot, X_val_director])

    y_train = train['Genre']
    y_val = validation['Genre']

    best_model = None
    best_kernel = None
    best_c = None
    best_f1 = 0  # To track the best F1 score
    f1_scores = {}  # Dictionary to store F1 scores for each kernel

    c_values = np.logspace(-4, 1, 20)  # Explore C values from 10^-4 to 10^1
    kernels = ['linear', 'rbf', 'poly']  # Define the kernels to test

    for kernel in kernels:
        print(f"\nTesting kernel: {kernel}")
        f1_scores[kernel] = []  # Initialize list for F1 scores of the current kernel
        
        for c in c_values:
            model = SVC(C=c, kernel=kernel, probability=True)  # Enable probability estimates

            # Train the model
            model.fit(X_train, y_train)

            # Get predicted probabilities
            predictions_proba = model.predict_proba(X_val)

            # Get predicted class indices by taking the index of the maximum value in predictions
            predicted_class_indices = np.argmax(predictions_proba, axis=1)

            # Map predicted indices back to genre labels
            unique_genres = np.unique(y_train)
            predictions = [unique_genres[idx] for idx in predicted_class_indices]

            # Calculate the F1 score between true labels and predicted labels
            current_f1 = f1_score(y_val, predictions, average='weighted')
            f1_scores[kernel].append(current_f1)  # Save the F1 score

            # If this F1 score is better than the best one, update best_c, best_f1, best_model, and best_kernel
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_c = c
                best_model = model
                best_kernel = kernel

            print(f"\tTested C={c:.4f}; F1 Score = {current_f1:.4f}")

    print(f"\nBest model is with kernel='{best_kernel}' and C={best_c:.4f} with F1 Score = {best_f1:.4f}")

    # Call the plot function to plot F1 vs C for the best kernel
    plot_f1_vs_c(c_values, f1_scores[best_kernel], best_kernel)

    return best_model, vectorizer, encoder


def plot_f1_vs_c(c_values, f1_scores, kernel):
    """
    Plots the F1 scores as a function of C values for the specified kernel.

    Args:
        c_values (list or array-like): The list of C values.
        f1_scores (list or array-like): The list of F1 scores corresponding to the C values.
        kernel (str): The kernel used in the SVM model.

    Returns:
        None: Displays the plot.
    """
    plt.figure()
    plt.plot(c_values, f1_scores)
    plt.title(f'F1 Score vs. C values for kernel: {kernel}')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('F1 Score')
    plt.xscale('log')  # Use logarithmic scale for better visibility
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../plots/svm_f1_vs_c.png")


def test_model(model, vectorizer, encoder, test):
    """
    Tests the trained SVM model on a new dataset with both plot and director features.

    Args:
        model: The trained SVM model.
        vectorizer: The fitted TfidfVectorizer used during training.
        encoder: The fitted OneHotEncoder used during training.
        test (pd.DataFrame): The test dataset with 'Plot' and 'Director' columns.

    Returns:
        np.ndarray: Predicted probabilities for each genre.
    """
    # Use the vectorizer and encoder trained during the model training
    X_plot = vectorizer.transform(test['Plot'])
    X_director = encoder.transform(test[['Director']])

    # Combine plot and director features
    X = hstack([X_plot, X_director])

    return model.predict_proba(X)
