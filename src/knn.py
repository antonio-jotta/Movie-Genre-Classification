from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt


def train_model(train, validation):
    """
    Trains a k-NN model using TfidfVectorizer for plots and OneHotEncoder for directors.

    Args:
        train (pd.DataFrame): Training dataset with 'Plot', 'Director', and 'Genre' columns.
        validation (pd.DataFrame): Validation dataset with 'Plot', 'Director', and 'Genre' columns.

    Returns:
        best_model: The best trained k-NN model.
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

    # Combine plot and director features using sparse matrix stacking
    X_train = hstack([X_train_plot, X_train_director])
    X_val = hstack([X_val_plot, X_val_director])

    y_train = train['Genre']
    y_val = validation['Genre']

    # Get the unique genres in the order that corresponds to the model's internal representation
    unique_genres = np.unique(y_train)

    best_k = None
    best_f1 = 0  # To track the best F1 score
    best_model = None
    f1_scores = []  # List to store F1 scores for each k

    k_values = range(1, 101)  # Loop through k values from 1 to 100
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)

        # Train the model
        model.fit(X_train, y_train)

        # Get predicted probabilities
        predictions_proba = model.predict_proba(X_val)

        # Convert probabilities to predicted class indices using argmax
        predicted_class_indices = np.argmax(predictions_proba, axis=1)

        # Map predicted indices back to genre labels using unique_genres
        predictions = [unique_genres[idx] for idx in predicted_class_indices]

        # Calculate the F1 score between true labels and predicted labels
        current_f1 = f1_score(y_val, predictions, average='weighted')
        f1_scores.append(current_f1)  # Save the F1 score

        # If this F1 score is better than the best one, update best_k and best_f1
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_k = k
            best_model = model

        print(f"Tested k={k}: F1 Score = {current_f1}")

    print(f"\nBest k={best_k} with F1 Score = {best_f1}")

    # Call the plot function to plot F1 vs k
    plot_f1_vs_k(k_values, f1_scores)

    return best_model, vectorizer, encoder


def plot_f1_vs_k(k_values, f1_scores):
    """
    Plots the F1 scores as a function of k values.

    Args:
        k_values (list or array-like): The list of k values.
        f1_scores (list or array-like): The list of F1 scores corresponding to the k values.

    Returns:
        None: Displays the plot.
    """
    plt.figure()
    plt.plot(k_values, f1_scores)
    plt.title('F1 Score vs. k values')
    plt.xlabel('k')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../plots/knn_f1")


def test_model(model, vectorizer, encoder, test):
    """
    Tests the trained k-NN model on a new dataset with both plot and director features.

    Args:
        model: The trained k-NN model.
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
