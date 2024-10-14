import pandas as pd
import re
from unidecode import unidecode
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack


# Download stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))


def split_data(train_data):
    """
    Splits the training data into training, validation, and test datasets.

    Args:
        train_data (pd.DataFrame): The original training data.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets (train_data, validation_data, test_data).
    """
    # First, split off 80% for training and 20% for validation + test
    train_data, validation_data = train_test_split(
        train_data, 
        test_size=0.2, 
        shuffle=True,
        random_state=42
    )

    return train_data, validation_data


def vectorize_and_encode_data(train, validation, test):
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

    # Use the vectorizer and encoder trained during the model training
    X_plot = vectorizer.transform(test['Plot'])
    X_director = encoder.transform(test[['Director']])

    # Combine plot and director features
    X_test = hstack([X_plot, X_director])

    return X_train, y_train, X_val, y_val, X_test


def check_imbalance(data):
    """
    Checks for class imbalance in the dataset based on the 'Genre' column and prints the count of each unique genre.

    Args:
        data (pd.DataFrame): Dataset containing a 'Genre' column.

    Returns:
        None: Prints the counts of each genre.
    """
    # Get the unique genres and their counts
    genre_counts = data['Genre'].value_counts()

    # Print the count of each genre
    print("Genre counts:")
    for genre, count in genre_counts.items():
        print(f"\t{genre}: {count} occurrences")


class DataCleaner:
    def __init__(self, train_path, test_path, output_path):
        """
        Initialize the DataCleaner with file paths.
        
        Args:
            train_path (str): Path to the training data file.
            test_path (str): Path to the test data file.
            output_path (str): Directory where output files will be saved.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path

    def read_data(self):
        """Reads the training and test datasets and cleans the data.

        Returns:
            tuple: A tuple containing the cleaned training dataset (pd.DataFrame) and the test dataset (pd.DataFrame).
        """
        # Read the training and test data
        train = pd.read_csv(self.train_path, delimiter='\t', names=["Title", "Industry", "Genre", "Director", "Plot"])
        test = pd.read_csv(self.test_path, delimiter='\t', names=["Title", "Industry", "Director", "Plot"])

        # Check for non-alphanumeric characters
        self.is_there_non_alphanum(train)

        # Clean all columns in the training data
        train = train.map(self.remove_non_alphanum)

        # Remove stop words in the 'Plot' column in the training data
        train['Plot'] = train['Plot'].apply(self.lowercase_and_remove_stopwords)

        # Output the cleaned training data to a txt file with tabs
        train.to_csv(f"{self.output_path}/cleaned_train_data.txt", sep='\t', header=False, index=False)

        return train, test

    def is_there_non_alphanum(self, dataset):
        """Finds and logs non-alphanumeric characters in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to check for non-alphanumeric characters.
        """
        # Extract non-alphanumeric characters from all columns
        non_alphanumeric_characters = set()
        for col in dataset.columns:
            for string in dataset[col]:
                if isinstance(string, str):  # Only process string columns
                    non_alphanumeric_characters.update(re.findall(pattern=r'[^a-zA-Z0-9\s]', string=string))

        # Output the non-alphanumeric characters to a txt file
        with open(f"{self.output_path}/non_alphanumeric_characters_found.txt", "w") as f:
            for char in sorted(non_alphanumeric_characters):
                f.write(f"{char}\n")

    def remove_non_alphanum(self, string):
        """Removes accents and non-alphanumeric characters from a string.

        Args:
            string (str): The string to be cleaned.

        Returns:
            str: The cleaned string without accents and non-alphanumeric characters.
        """
        if isinstance(string, str):  # Ensure only strings are processed
            string = unidecode(string)  # Remove accents
            string = re.sub(r'[^a-zA-Z0-9\s]', '', string)  # Remove non-alphanumeric characters
        return string

    def lowercase_and_remove_stopwords(self, string):
        """Removes stop words from a given string.

        Args:
            string (str): The input string from which stop words are to be removed.

        Returns:
            str: The processed string with stop words removed and converted to lowercase.
        """
        # Convert text to lowercase and split into words
        words = string.lower().split()
        # Remove stop words
        filtered_words = [word for word in words if word not in stop_words]
        # Join the filtered words back into a single string
        return ' '.join(filtered_words)
