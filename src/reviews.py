import mgc_input
import mgc_train
import mgc_test
import mgc_output


def main():
    """
    Main function to execute the movie genre classification pipeline.

    Returns:
        None: This function does not return any value.
    """
    # Initialize the DataCleaner with paths for training and test data
    Cleaner = mgc_input.DataCleaner(
        train_path="../data/train.txt", 
        test_path="../data/test_no_labels.txt", 
        output_path="../output"
    )
    
    # Read and clean the training and test data
    train_data, results_test_data = Cleaner.read_data()

    # Split the training data into training (80%) and validation/test sets (10% each)
    train_data, validation_data, test_data = mgc_input.split_data(train_data=train_data)

    # # Uncomment the following line if the model needs to be trained again
    # mgc_train.train_model(train=train_data, validation=validation_data) 

    # Evaluate the model's accuracy on the test data
    mgc_test.accuracy_in_test_data(train_data=train_data, test_data=test_data)

    # Uncomment the following lines to get predictions on the results test data
    results_test_predictions = mgc_test.test_sample(test=results_test_data)

    # Print the results of the predictions on the results test data
    mgc_output.print_result(
        data=results_test_data, 
        predictions=results_test_predictions, 
        unique_genres=train_data['Genre'].unique()
    )


if __name__ == "__main__":
    main()
