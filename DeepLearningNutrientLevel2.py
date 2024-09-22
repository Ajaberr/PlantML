import numpy as np #numpy
import copy         #copy
import matplotlib.pyplot as plt #matplot for graphs
import scipy #helps numpy
from PIL import Image  #helps with dealing with images in python
from scipy import ndimage  #helps image processing
import os


training_Y_1 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap2"
training_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap1"
training2_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap3"
training3_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap4"

test_Y_1 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap2"
test_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap1"
test2_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap3"
test3_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap4"


# First, store functions in arrays:

def  load_images_and_labels(folder_paths, label):
    images = []
    labels = []

    for folder_path in folder_paths:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                img_path = os.path.join(folder_path, file_name)
                img = Image.open(img_path).convert("RGB")
                img = img.resize((100, 100))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)



# Load the training data
X_train_1, Y_train_1 = load_images_and_labels([training_Y_1], 1)
X_train_0, Y_train_0 = load_images_and_labels([training_Y_0, training2_Y_0, training3_Y_0], 0)

# Load the test data
X_test_1, Y_test_1 = load_images_and_labels([test_Y_1], 1)
X_test_0, Y_test_0 = load_images_and_labels([test_Y_0, test2_Y_0, test3_Y_0], 0)

# Combine the data
X_train = np.concatenate((X_train_1, X_train_0), axis=0)
Y_train = np.concatenate((Y_train_1, Y_train_0), axis=0)

X_test = np.concatenate((X_test_1, X_test_0), axis=0)
Y_test = np.concatenate((Y_test_1, Y_test_0), axis=0)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the images to be one-dimensional (if using logistic regression or a simple neural network)
X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

# Ensure labels are the correct shape
Y_train = Y_train.reshape(1, -1)
Y_test = Y_test.reshape(1, -1)

# Ensures random function calls are consistent
np.random.seed(1)


# Functions for the mathematical functions
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def sigmoid_backward(dA, activation_cache):
    A, Z = sigmoid(activation_cache)
    return dA * A * (1 - A)

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)  # Copy the gradient to avoid modifying the original data
    dZ[Z <= 0] = 0  # Set gradients to 0 where Z <= 0
    return dZ

#Initialize parameters

input_layer = np.array([np.squeeze(X_train_flatten.shape[0])]) # array containing number of input nodes
hidden_layer_dims = np.ones((1)).astype(int) * 5 # (30,) element array containing number of nodes for each hidden layer
output_layer_dims = np.array([1]) # number of output nodes
layer_dims = np.concatenate((input_layer, hidden_layer_dims, output_layer_dims), axis = 0)


def initialize_parameters(layer_dims): #layer_dims length = 1 + number of layers ==> as it starts at n[0] (input layer) but finishes and n[L] (output layer)
    np.random.seed(3)
    parameters = { 

    }
    for l in range(1, len(layer_dims)):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b): 
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache

def linear_activation_forward(A_prev, W, b, activation): #Use activation function to make the forward function
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2

    for l in range(1, L): # From hidden layer 1 to last hidden layer = relu  activation
        A, cache = linear_activation_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True )
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache_of_current_layer, activation):
    linear_cache, activation_cache = cache_of_current_layer

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape((AL.shape))
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")

    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)): #Loop from L-2 to 0
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters)//2

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters


    

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations = 3000, print_cost = True):

    np.random.seed(1)

    costs = []
    parameters = initialize_parameters(layer_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration", i,":", np.squeeze(cost))
            costs.append(cost)
    return parameters, costs
    
def predict(X, parameters):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    AL, caches = L_model_forward(X, parameters)

    for i in range(AL.shape[1]):
        if AL[0, i] > 0.5:
            Y_prediction[0,i] = 1.0
        else:
            Y_prediction[0,i] = 0.0
    return Y_prediction, AL

parameters, costs = L_layer_model(X_train_flatten, Y_train, layer_dims, learning_rate = 0.005, num_iterations = 3000, print_cost = True)
Y_prediction_test, ALTest = predict(X_test_flatten, parameters)
Y_prediction_train, ALTrain = predict(X_train_flatten, parameters)
print("train accuracy:", (100-np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy:", (100-np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


# 99 percent accuracy

import csv

with open('Level2.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    for prediction in Y_prediction_test[0]: 
        writer.writerow([prediction])
    
    
    writer.writerow([100-np.mean(np.abs(Y_prediction_test - Y_test)) * 100])

    for prediction in ALTest[0]:  
        writer.writerow([prediction])