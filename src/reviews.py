import mgc_input
# import bert_train
# import bert_test
import mgc_output
import knn
import svm


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
    mgc_input.check_imbalance(data=train_data)

    # Split the training data into training (80%) and validation/test sets (10% each)
    train_data, validation_data, test_data = mgc_input.split_data(train_data)

    ####################
    # Predict using BERT
    ####################
    # # Uncomment the following line if the model needs to be trained again
    model = bert_train.train_model(train_data, validation_data) 

    # Evaluate the model's accuracy on the test data
    bert_test.accuracy_in_test_data(train_data, test_data, model)

    # Uncomment the following lines to get predictions on the results test data
    results_test_predictions = bert_test.test_sample(results_test_data, model)

    ####################
    # Predict using k-NN
    ####################
    # X_train, y_train, X_val, y_val, X_test = mgc_input.vectorize_and_encode_data(
    #     train=train_data,
    #     validation=test_data,
    #     test=results_test_data
    # )
    # knn_model, k_values, acc_scores = knn.train_model(
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_val=X_val,
    #     y_val=y_val
    # )
    # knn.plot_acc_vs_k(k_values=k_values, acc_scores=acc_scores)
    # results_test_predictions = knn.test_model(model=knn_model, test=X_test)

    ###################
    # Predict using SVM
    ###################
    #X_train, y_train, X_val, y_val, X_test = mgc_input.vectorize_and_encode_data(
    #    train=train_data,
    #    validation=validation_data,
    #    test=results_test_data
    #)
    ## # Do grid search to get the best parameters
    ## svm_model, results = svm.do_svm_grid_search(
    ##     X_train=X_train,
    ##     y_train=y_train,
    ##     X_val=X_val,
    ##     y_val=y_val
    ## )
    ## svm.plot_svm_history(grid_search_results=results)
    #
    ## Use the best found model in grid search
    #svm_model = svm.svm_model(
    #    X_train=X_train, 
    #    y_train=y_train,
    #    X_val=X_val,
    #    y_val=y_val
    #)
    #results_test_predictions = svm.test_model(model=svm_model, test=X_test)

    ###############################################################
    # Print the results of the predictions on the results test data
    ###############################################################
    mgc_output.print_result(
        data=results_test_data, 
        predictions=results_test_predictions, 
        unique_genres=train_data['Genre'].unique()
    )


if __name__ == "__main__":
    main()
