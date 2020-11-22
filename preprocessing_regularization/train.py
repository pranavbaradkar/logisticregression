import numpy as np
import math 
import csv

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


def sigmoid(Z):

  return 1/(1+np.exp(-Z))



def compute_gradient_of_cost_function(X, Y, W, b,iteration,learning_rate,Lambda):
    for i in range(iteration):
        Z = np.dot(X,W) + b 
        A = sigmoid(Z) 
        n = len(Y) 
        dz = A - Y
        dw = (1/n)*np.dot(X.T,dz)
        db = (1/n)*np.sum(dz) 
        regularization = (learning_rate * Lambda / n) * W
        W = (W - regularization) - learning_rate * dw
        b = b - learning_rate * db
        mse = Lambda * np.sum(np.square(W))/2
        cost = (-1/n) * (np.sum(np.multiply(Y , np.log(A)) + np.multiply((1-Y) , np.log(1-A))) - mse)
        print(cost,i)
    return W,b
    
    
    
if __name__ == "__main__":
    column_indices = np.array([2,5])
    weight = []
    be = []
    X = np.genfromtxt("train_X_pr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    X = replace_null_values_with_mean(X)
    X = mean_normalize(X, column_indices)
    Y = np.genfromtxt("train_Y_pr.csv", delimiter=',', dtype=np.float64)
    for i in range(1,2):
        W = np.zeros((X.shape[1],1))
        Y1 = np.where(Y==i,1,0)
        Y1 = Y1.reshape((len(Y1),1))
        b = 5
        W, b = compute_gradient_of_cost_function(X, Y1, W, b,50000,1,0.2)
        W = np.round(W, 3).T.tolist()[0]
        b = np.round(b, 3)
        weight.append(W) 
        be.append(b)
print(weight)
print(be)
with open("WEIGHTS1_FILE.csv", 'w', newline='') as csv_file:
    wr = csv.writer(csv_file)
    wr.writerows(weight)
    csv_file.close()
