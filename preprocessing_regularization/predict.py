import numpy as np
import csv
import sys

from validate import validate

def replace_null_values_with_mean(X):
    col_mean = np.round(np.nanmean(X, axis=0), 3)

    #Find indicies that we need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X

def mean_normalize(X, column_indices):
    col_mean = np.mean(X,axis = 0)
    col_min = np.min(X,axis = 0)
    col_max = np.max(X,axis=0)
    col_std = np.std(X,axis=0)
    for i in range(len(X)):
        for j in column_indices:
            X[i][j] = (X[i][j] - col_mean[j]) / (col_max[j]-col_min[j])
    
    return X

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
def sigmoid(Z):

    return 1/(1+np.exp(-Z))

def import_data_and_weights(test_X_file_path, weights_file_path):
    X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    column_indices = np.array([2,5])
    X = replace_null_values_with_mean(X)
    X = mean_normalize(X, column_indices)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return X, weights


def predict_target_values(test_X, weights):
    test_Y = []
    b = [-1.075,1.075]
    for test in test_X:
        h = []
        i = 0
        for weight in weights:
            h.append(sigmoid(np.dot(test,weight.T)+b[i]))
            i = i+1
        maximum = np.max(h,axis=0) 
        label = h.index(maximum)
        test_Y.append([label])
    return test_Y 
    
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.

    """
    Note:
    The preprocessing techniques which are used on the train data, should also be applied on the test 
    1. The feature scaling technique used on the training data should be applied as it is (with same mean/standard_deviation/min/max) on the test data as well.
    2. The one-hot encoding mapping applied on the train data should also be applied on test data during prediction.
    3. During training, you have to write any such values (mentioned in above points) to a file, so that they can be used for prediction.
     
    You can load the weights/parameters and the above mentioned preprocessing parameters, by reading them from a csv file which is present in the SubmissionCode.zip
    """
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in regularization.zip and imported properly.
    """

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    pred_Y = np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 