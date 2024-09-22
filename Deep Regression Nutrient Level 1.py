import numpy as np #numpy
import copy         #copy
import matplotlib.pyplot as plt #matplot for graphs
import scipy #helps numpy
from PIL import Image  #helps with dealing with images in python
from scipy import ndimage  #helps image processing
import os
import csv



training_Y_1 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap1"
training_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap2"
training2_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap3"
training3_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Training\swap4"

test_Y_1 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap1"
test_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap2"
test2_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap3"
test3_Y_0 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap4"

# Define a function to load images and their labels
def load_images_and_labels(folder_paths, label):
    images = []
    labels = []
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # add more image formats if needed
                img_path = os.path.join(folder_path, filename)
                image = Image.open(img_path).convert('RGB')
                image = image.resize((100, 100))  # resize to a consistent size = resize method of PIL library and only changes height and width dimensions.
                images.append(image)
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



"""
import matplotlib.pyplot as plt

def display_image(image_array, title="Image"):
    plt.imshow(image_array)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example of displaying an image
display_image(X_train_1[0], "First Training Image with Label 1")
display_image(X_train_0[0], "First Training Image with Label 0")

//Image and allows pointer to read array values of every pixel.
"""








def sigmoid(z):
    return 1/(1 + np.exp(-z))

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)

    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A + np.exp(-8)))

    dw = (1/m) * np.dot(X, (A-Y).T)

    db = (1/m) * np.sum(A-Y)

    cost = np.squeeze(np.array(cost)) #Ensures cost is 1D Array (1,)

    grads = { "dw": dw,
              "db": db
              }
    return grads, cost


def optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False):
   
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b  - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            
            if print_cost:
                print("Cost After Iteration:", cost)

    params = {"w": w,
              "b": b
              }

    grads = {"dw" : dw,
             "db" : db
             }
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1.0
        else:
            Y_prediction[0,i] = 0.0
    return Y_prediction, A


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = np.zeros((X_train.shape[0], 1)), 0

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]
    Y_prediction_test, ALTest = predict(w, b, X_test)
    Y_prediction_train, ALTrain = predict(w, b, X_train)

    if print_cost:
        print("train accuracy:", (100-np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy:", (100-np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs":costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "ALTest": ALTest,
         "num_iterations" : num_iterations,
         "test accuracy": (100-np.mean(np.abs(Y_prediction_test - Y_test)) * 100)
         }
    
    return d


logistic_regression_model = model(X_train_flatten, Y_train, X_test_flatten, Y_test, num_iterations = 10000, learning_rate = 0.001, print_cost = True)


"""
learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print("A training model with learning rate:", lr)
    models[str(lr)] = model(X_train_flatten, Y_train, X_test_flatten, Y_test, num_iterations = 2000, learning_rate = lr, print_cost = False)
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label = str(models[str(lr)]["learning_rate"]))
plt.ylabel("cost")
plt.xlabel("iterations(hundreds)")
legend = plt.legend(loc = "upper center", shadow = True)
frame = legend.get_frame()
frame.set_facecolor("0.9")
plt.show()    


Hence peak learning rate is 0.001, and num_iterations = 2000 is suitable
"""

# 98 percent - 10000 num


import csv

with open('Level1.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    for prediction in logistic_regression_model["Y_prediction_test"][0]:  
        writer.writerow([prediction])
    
    
    writer.writerow([logistic_regression_model["test accuracy"]])

    for prediction in logistic_regression_model["ALTest"][0]:  
        writer.writerow([prediction])



  