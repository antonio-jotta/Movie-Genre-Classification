import torch
import numpy as np
from transformers import DistilBertTokenizer
from sklearn.metrics import f1_score
import ctypes

def trim_memory():
  libc = ctypes.CDLL("libc.so.6")
  return libc.malloc_trim(0)

def test_sample(test, model):
    """
    Makes predictions on the provided test data using the provided model.

    Args:
        test (pd.DataFrame): The test dataset containing the 'Plot' and 'Director' columns for predictions.
        model: The trained model for making predictions.

    Returns:
        np.ndarray: The predicted probabilities for each genre based on the input plots and directors.
    """
    # Convert the 'Plot' and 'Director' columns to lists
    X_test = test['Plot'].tolist()
    directors_test = test['Director'].tolist()

    # Set the model to evaluation mode
    model = model.to("cpu")
    model.eval()

    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[DIRECTOR]']})

    trim_memory()
    # Tokenize the test inputs with the director information
    with torch.no_grad():  # Disable gradient calculation
        director_names = [f"[DIRECTOR] {director}" for director in directors_test]
        texts_with_directors = [f"{director_name} {text}" for director_name, text in zip(director_names, X_test)]
        text_inputs = tokenizer(texts_with_directors, padding=True, truncation=True, return_tensors='pt', max_length=512)

        # Move inputs to CPU
        input_ids = text_inputs['input_ids'].to("cpu")
        attention_mask = text_inputs['attention_mask'].to("cpu")

        # Make predictions using the model
        print("calling model test")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply softmax to get probabilities
        predictions = torch.softmax(outputs['logits'], dim=1)

    return predictions.cpu().numpy()  # Convert to numpy array and move to CPU if necessary



def accuracy_in_test_data(train_data, test_data, model):
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
    test_predictions = test_sample(test_data, model)

    # Interpret the predictions against the true genre labels
    interpret_predictions_test_data(
        genre_labels=genre_column, 
        predictions=test_predictions, 
        unique_genres=train_data['Genre'].unique()
    )

def interpret_predictions_test_data(genre_labels, predictions, unique_genres):
    """
    Interprets the model's predictions, calculates the accuracy against true labels, 
    and computes the F1 score.

    Args:
        genre_labels (pd.Series): The true genre labels from the test dataset.
        predictions (np.ndarray): The predicted probabilities for each genre.
        unique_genres (list): A list of unique genre labels for decoding predictions.

    Returns:
        None: Prints the number of matches, percentage accuracy, and the F1 score.
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

    # Calculate and print the F1 score
    f1 = f1_score(genre_labels, predicted_classes, average='weighted')
    print(f"F1 Score: {f1:.4f}")
