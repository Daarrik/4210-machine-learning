#-------------------------------------------------------------------------
# AUTHOR: Darrik Houck
# FILENAME: perceptron.py
# SPECIFICATION: Iterates through parameters and applies them to a Perceptron classifier
# and a Multilayer Perceptron classifier. Tracks the highest accuracy for both classifiers
# with the corresponding parameters.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test
highest_acc_per = 0
highest_acc_mlp = 0
for w in n: #iterates over n

    for b in r: #iterates over r

        for a in range(2): #iterates over the algorithms
            #Create a Neural Network classifier
            if a==0:
               clf = Perceptron(eta0=w, random_state=b, max_iter=1000) #eta0 = learning rate, random_state = shuffle the training data
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,), random_state=b, max_iter=1000) #learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer, random_state = shuffle the training data

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            num_correct = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
               prediction = clf.predict([x_testSample])
               if prediction == y_testSample:
                  num_correct += 1
            accuracy = num_correct/len(X_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, random_state=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, random_state=False"
            #--> add your Python code here
            if a==0:
               if accuracy > highest_acc_per:
                  highest_acc_per = accuracy
                  print(f'Highest Perceptron accuracy so far: {highest_acc_per}, Parameters: learning rate={w}, random_state={b}')
            else:
               if accuracy > highest_acc_mlp:
                  highest_acc_mlp = accuracy
                  print(f'Highest MLP accuracy so far: {highest_acc_mlp}, Parameters: learning rate={w}, random_state={b}')
