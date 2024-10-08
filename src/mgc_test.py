import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np


def test_sample(test):
    """
    Loads a pre-trained model and makes predictions on the provided test data.

    Args:
        test (pd.DataFrame): The test dataset containing the 'Plot' column for predictions.

    Returns:
        np.ndarray: The predicted probabilities for each genre based on the input plots.
    """
    # Convert the 'Plot' column to a list
    X_test = test['Plot'].tolist()
    
    # Load the pre-trained model
    loaded_model = tf.keras.models.load_model('trained_models/bert_en_uncased')

    # Make predictions using the loaded model
    predictions = loaded_model.predict(tf.constant(X_test))
    
    return predictions


def accuracy_in_test_data(train_data, test_data):
    """
    Calculates and interprets the accuracy of predictions on the test data.

    Args:
        train_data (pd.DataFrame): The training dataset used for model training.
        test_data (pd.DataFrame): The test dataset containing the 'Genre' column for evaluation.

    Returns:
        None: This function does not return any value; it prints accuracy results.
    """
    # Save the "Genre" column into a separate variable
    genre_column = test_data['Genre']

    # Drop the "Genre" column from test_data for predictions
    test_data = test_data.drop(columns=['Genre'])

    # Get predictions for the test data
    test_predictions = test_sample(test=test_data)

    # Interpret the predictions against the true genre labels
    interpret_predictions_test_data(
        genre_labels=genre_column, 
        predictions=test_predictions, 
        unique_genres=train_data['Genre'].unique()
    )
    

def interpret_predictions_test_data(genre_labels, predictions, unique_genres):
    """
    Interprets the model's predictions and calculates the accuracy against true labels.

    Args:
        genre_labels (pd.Series): The true genre labels from the test dataset.
        predictions (np.ndarray): The predicted probabilities for each genre.
        unique_genres (list): A list of unique genre labels for decoding predictions.

    Returns:
        None: This function does not return any value; it prints the number of matches and percentage.
    """
    # Get the predicted class indices by taking the index of the maximum value in predictions
    predicted_class_indices = np.argmax(predictions, axis=1)

    # Create a mapping from genre names to indices
    genre_to_index = {genre: idx for idx, genre in enumerate(unique_genres)}
    index_to_genre = {idx: genre for genre, idx in genre_to_index.items()}  # For decoding

    # Decode predicted indices to genre labels
    predicted_classes = [index_to_genre[idx] for idx in predicted_class_indices]

    # Count the number of matches between predicted_classes and genre_labels
    matches = sum(predicted_class == label for predicted_class, label in zip(predicted_classes, genre_labels))
    
    # Print the number of matches out of the number of pairs analyzed
    total_pairs = len(genre_labels)

    # Calculate the percentage of matches
    percentage_matches = (matches / total_pairs) * 100 if total_pairs > 0 else 0

    # Print the number of matches and percentage
    print(f"Number of matches: {matches}/{total_pairs} ({percentage_matches:.2f}%).")
