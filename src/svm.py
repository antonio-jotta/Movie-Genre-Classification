from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns


def do_svm_grid_search(X_train, y_train, X_val, y_val):
    """
    Trains an SVM model using GridSearchCV to tune the C, kernel, degree, and gamma parameters.
    It tests multiple kernels and returns the best model with its optimal hyperparameters.

    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        X_val: Feature matrix for validation.
        y_val: Labels for validation.

    Returns:
        best_model: The best trained SVM model found by GridSearchCV.
        grid_search: The GridSearchCV object containing the best model and other information.
    """
    
    param_grid = [
        {'kernel': ['linear'], 
         'C': [0.001, 0.1, 1, 10, 100, 1000]
        },
        {'kernel': ['rbf', 'sigmoid'], 
         'C': [0.001, 0.1, 1, 10, 100, 1000], 
         'gamma': ['scale']
        },
        {'kernel': ['poly'], 
         'C': [0.001, 0.1, 1, 10, 100, 1000], 
         'degree': range(1, 7), 
         'gamma': ['scale']
        }
    ]
    
    # Create an SVM classifier
    model = SVC(probability=True, random_state=42)  # Enabling probability estimation
    
    # Set up GridSearchCV with cross-validation and scoring
    # grid_search = GridSearchCV(
    #     estimator=model, 
    #     param_grid=param_grid, 
    #     scoring='f1_weighted', 
    #     cv=5,  # 5-fold cross-validation
    #     verbose=2,  # Set to higher values for more output (e.g., verbose=1 or 2)
    #     n_jobs=-1,  # Use all available cores
    #     refit=True
    # )
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=5,  # 5-fold cross-validation
        verbose=2,  # Set to higher values for more output (e.g., verbose=1 or 2)
        n_jobs=-1,  # Use all available cores
        refit=True
    )
    
    # Perform the grid search on the training data
    grid_search.fit(X=X_train, y=y_train)
    
    # Best model from the grid search
    best_model = grid_search.best_estimator_

    # Validation on the held-out validation set
    predictions_proba = best_model.predict_proba(X=X_val)

    # Get predicted class indices by taking the index of the maximum value in predictions
    predicted_class_indices = np.argmax(predictions_proba, axis=1)

    # Map predicted indices back to genre labels
    unique_genres = np.unique(y_train)
    predictions = [unique_genres[idx] for idx in predicted_class_indices]

    # Calculate the F1 score for the validation set
    best_f1 = f1_score(y_true=y_val, y_pred=predictions, average='weighted')
    # Calculate the accuracy for the validation set
    best_accuracy = accuracy_score(y_true=y_val, y_pred=predictions)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1 score on validation set: {best_f1}")
    print(f"Best Accuracy on validation set: {best_accuracy}")

    np.save('../output/grid_search_results.npy', grid_search.cv_results_)
    
    return best_model, grid_search.cv_results_


def svm_model(X_train, y_train, X_val, y_val):
    """
    Train an SVM model using the best parameters from grid search and evaluate it on the validation set.

    Args:
        X_train (numpy.ndarray): Training input features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation input features.
        y_val (numpy.ndarray): Validation labels.

    Returns:
        None
    """
    best_params = {
        'kernel': 'sigmoid',
        'C': 10,
        'gamma': 'scale'
    }

    # Create the SVM model with the best parameters
    svm_model = SVC(
        kernel=best_params['kernel'], 
        C=best_params['C'], 
        gamma=best_params.get('gamma', 'scale'),
        # verbose=True,
        probability=True,
        random_state=42
    )

    # Train the SVM model
    svm_model.fit(X=X_train, y=y_train)

    # Validation on the held-out validation set
    predictions_proba = svm_model.predict_proba(X=X_val)

    # Get predicted class indices by taking the index of the maximum value in predictions
    predicted_class_indices = np.argmax(predictions_proba, axis=1)

    # Map predicted indices back to genre labels
    unique_genres = np.unique(y_train)
    predictions = [unique_genres[idx] for idx in predicted_class_indices]

    # Calculate the F1 score for the validation set
    best_f1 = f1_score(y_true=y_val, y_pred=predictions, average='weighted')
    # Calculate the accuracy for the validation set
    best_accuracy = accuracy_score(y_true=y_val, y_pred=predictions)

    print(f"Best parameters found: {best_params}")
    print(f"Best F1 score on validation set: {best_f1}")
    print(f"Best Accuracy on validation set: {best_accuracy}")

    do_confusion_matrix(y_true=y_val, y_pred=predictions)

    return svm_model


def do_confusion_matrix(y_true, y_pred):
    # Generate confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Display the confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("../plots/svm_cm")


def plot_svm_history(grid_search_results):
    """
    Plots the accuracy score as a function of regularization parameter C for different SVM kernels.

    Args:
        grid_search_results (dict): The results from a grid search, typically the `cv_results_` from a 
                                    GridSearchCV object. It should include 'param_C', 'param_kernel', 
                                    and 'mean_test_score' among other fields.

    Returns:
        None: Displays the plot showing the relationship between C, Acc score, and kernel.
    """
    # # Load the cv_results_ from the .npy file
    # cv_results = np.load('../output/grid_search_results.npy', allow_pickle=True).item()
    # grid_search_results = np.load('../output/grid_search_results.npy', allow_pickle=True).item()

    # Convert grid search results to a pandas DataFrame for easier manipulation
    df_results = pd.DataFrame(grid_search_results)

    # Filter the relevant columns
    df_filtered = df_results[['param_C', 'param_kernel', 'mean_test_score']]

    # Get the unique SVM kernels used in the grid search
    kernels = df_filtered['param_kernel'].unique()

    # Loop through each kernel
    for kernel in kernels:
        # Filter the DataFrame for the current kernel
        df_kernel = df_filtered[df_filtered['param_kernel'] == kernel]
        
        # Plot accuracy score as a function of the regularization parameter C
        plt.plot(df_kernel['param_C'], df_kernel['mean_test_score'], label=kernel)

    # Customize the plot
    plt.xlabel('C (Regularization parameter)')
    plt.ylabel('Accuracy')
    plt.legend()  # Add a legend to distinguish between kernels
    plt.xscale('log')  # Use logarithmic scale for C, as SVM parameters often vary across orders of magnitude
    plt.grid(True)  # Add gridlines for better readability
    plt.savefig("../plots/svm_acc_vs_c")


def test_model(model, test):
    """
    Tests the trained SVM model on a new dataset with both plot and director features.

    Args:
        model: The trained SVM model.
        test (pd.DataFrame): The test dataset with 'Plot' and 'Director' columns.

    Returns:
        np.ndarray: Predicted probabilities for each genre.
    """
    return model.predict_proba(test)