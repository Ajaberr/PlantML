import copy         #copy
import matplotlib.pyplot as plt #matplot for graphs
import scipy #helps numpy
from PIL import Image  #helps with dealing with images in python
from scipy import ndimage  #helps image processing
import os
import csv
import numpy as np
test_Y_1 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap1"
test_Y_2 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap2"
test_Y_3 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap3"
test_Y_4 = r"C:\Users\ahmed\Documents\Machine Learning\Projects\Simple Project 1 - NN Logistic Regression for Plant Deficiency\NitrogenDeficiencyImage\Test\swap4"

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
X_test_1, Y_test_1 = load_images_and_labels([test_Y_1], 1)
X_test_2, Y_test_2 = load_images_and_labels([test_Y_2], 2)
X_test_3, Y_test_3 = load_images_and_labels([test_Y_3], 3)
X_test_4, Y_test_4 = load_images_and_labels([test_Y_4], 4)

Y_test = np.concatenate((Y_test_1, Y_test_2, Y_test_3, Y_test_4), axis = 0)
Y_pred = []      


with open('Level1.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\n')
    Level1 = []
    for row in csv_reader:
        Level1.append([float(item) for item in row])

with open('Level2.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\n')
    Level2 = []
    for row in csv_reader:
        Level2.append([float(item) for item in row])


with open('Level3.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\n')
    Level3 = []
    for row in csv_reader:
        Level3.append([float(item) for item in row])

with open('Level4.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\n')
    Level4 = []
    for row in csv_reader:
        Level4.append([float(item) for item in row])

print(np.squeeze(Level4[0])) # print float number of first element in array




accuracy_test1 = np.squeeze(Level1[len(Y_test)]) 
accuracy_test2 = np.squeeze(Level2[len(Y_test)]) 
accuracy_test3 = np.squeeze(Level3[len(Y_test)]) 
accuracy_test4 = np.squeeze(Level4[len(Y_test)]) 

'''
Weightage1 = np.squeeze(Level1[len(Y_test) + 1:])  * 2/3 * 100 + 1/3 * accuracy_test1
Weightage2 = np.squeeze(Level2[len(Y_test) + 1:])  * 2/3 * 100 + 1/3 * accuracy_test2
Weightage3 = np.squeeze(Level3[len(Y_test) + 1:])  * 2/3 * 100 + 1/3 * accuracy_test3
Weightage4 = np.squeeze(Level4[len(Y_test) + 1:])  * 2/3 * 100 + 1/3 * accuracy_test4
'''
#Doesn't Work, only AL works better better.
Weightage1 = np.squeeze(Level1[len(Y_test) + 1:])  
Weightage2 = np.squeeze(Level2[len(Y_test) + 1:])  
Weightage3 = np.squeeze(Level3[len(Y_test) + 1:])  
Weightage4 = np.squeeze(Level4[len(Y_test) + 1:])

for i in range(len(Y_test_1)):
    max_val = np.max([Weightage1[i], Weightage2[i + len(Y_test_2)], Weightage3[i + len(Y_test_3)], Weightage4[i + len(Y_test_4)]])
    if max_val == Weightage1[i]:
        Y_pred.append(1)
    elif max_val == Weightage2[i + len(Y_test_2)]:
        Y_pred.append(2)
    elif max_val == Weightage3[i + len(Y_test_3)]:
        Y_pred.append(3)
    elif max_val == Weightage4[i + len(Y_test_4)]:
        Y_pred.append(4)


for i in range(len(Y_test_2)):
    max_val = np.max([Weightage1[i + len(Y_test_1)], Weightage2[i], Weightage3[i + len(Y_test_3) + len(Y_test_1)], Weightage4[i + len(Y_test_4) + len(Y_test_1)]])
    if max_val == Weightage1[i + len(Y_test_1)]:
        Y_pred.append(1)
    elif max_val == Weightage2[i]:
        Y_pred.append(2)
    elif max_val == Weightage3[i + len(Y_test_3)]:
        Y_pred.append(3)
    elif max_val == Weightage4[i + len(Y_test_4)]: 
        Y_pred.append(4)

for i in range(len(Y_test_3)):
    max_val = np.max([Weightage1[i + len(Y_test_1) + len(Y_test_2)], Weightage2[i + len(Y_test_1) + len(Y_test_2)], Weightage3[i], Weightage4[i + len(Y_test_4) + len(Y_test_1) + len(Y_test_2)]])
    if max_val == Weightage1[i + len(Y_test_1) + len(Y_test_2)]:
        Y_pred.append(1)
    elif max_val == Weightage2[i + len(Y_test_1) + len(Y_test_2)]:
        Y_pred.append(2)
    elif max_val == Weightage3[i]:
        Y_pred.append(3)
    elif max_val == Weightage4[i + len(Y_test_4) + len(Y_test_1) + len(Y_test_2)]: 
        Y_pred.append(4)

for i in range(len(Y_test_4)):
    max_val = np.max([Weightage1[i + len(Y_test_1) + len(Y_test_2) + len(Y_test_3)], Weightage2[i + len(Y_test_1) + len(Y_test_2) + len(Y_test_3)], Weightage3[i + len(Y_test_1) + len(Y_test_2) + len(Y_test_3)], Weightage4[i]])
    if max_val == Weightage1[i + len(Y_test_1) + len(Y_test_2) + len(Y_test_3)]:
        Y_pred.append(1)
    elif max_val == Weightage2[i + len(Y_test_1) + len(Y_test_2) + len(Y_test_3)]:
        Y_pred.append(2)
    elif max_val == Weightage3[i + len(Y_test_1) + len(Y_test_2) + len(Y_test_3)]:
        Y_pred.append(3)
    elif max_val == Weightage4[i]: 
        Y_pred.append(4)











accuracy = (np.sum(np.array(Y_pred) == np.array(Y_test)) / len(Y_test)) * 100
print('Final Accuracy:',accuracy )


#95% Accuracy Rate Classification