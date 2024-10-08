import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np


def print_result(data, predictions, unique_genres):
    """
    Processes the predictions and saves the results to a text file.

    Args:
        data (pd.DataFrame): The original data containing movie titles.
        predictions (np.ndarray): The predicted probabilities for each genre.
        unique_genres (list): A list of unique genre labels corresponding to the predictions.

    Returns:
        None: This function does not return any value. It writes results to a file.
    """
    # Get the predicted class indices by taking the index of the maximum value in predictions
    predicted_class_indices = np.argmax(predictions, axis=1)

    # Create a mapping from genre names to indices
    genre_to_index = {genre: idx for idx, genre in enumerate(unique_genres)}
    index_to_genre = {idx: genre for genre, idx in genre_to_index.items()}  # For decoding

    # Decode predicted indices to genre labels
    predicted_classes = [index_to_genre[idx] for idx in predicted_class_indices]

    # Specify the output filename
    filename = "../output/results.txt"

    # Write the predicted genres along with their corresponding titles to the output file
    with open(filename, 'w+') as f:
        for i, predicted_class in enumerate(predicted_classes):
            f.write(f"{data['Title'][i]}: {predicted_class}\n")
