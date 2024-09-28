import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

def print_result(data, predictions, unique_genres):
    predicted_class_indices = np.argmax(predictions)

    genre_to_index = {genre: idx for idx, genre in enumerate(unique_genres)}
    index_to_genre = {idx: genre for genre, idx in genre_to_index.items()}  # For decoding
    predicted_class_indices = np.argmax(predictions, axis=1)
    
    predicted_classes = [index_to_genre[idx] for idx in predicted_class_indices]

    filename = "result.txt"

    with open(filename, 'w+') as f:
        for i, predicted_class in enumerate(predicted_classes):
            f.write(f"{data['Title'][i]}: {predicted_class}\n")