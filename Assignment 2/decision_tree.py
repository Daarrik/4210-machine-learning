#-------------------------------------------------------------------------
# AUTHOR: Darrik Houck
# FILENAME: decision_tree.py
# SPECIFICATION: Returns the lowest accuracy of 10 decision trees made from 
# 3 training sets (contact_lens_training_1.csv, contact_lens_training_2.csv), 
# contact_lens_training_3.csv) when tested against the test set contact_lens_test.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv
import os
os.chdir("C:/Users/Darrik/Documents/Python/4210/Assignment 2")

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Dictionaries for training feature transformation
age = {"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3}
sp = {"Myope": 1, "Hypermetrope": 2}
ast = {"Yes": 1, "No": 2}
tpr = {"Normal": 1, "Reduced": 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    for row in dbTraining:
        X.append([age[row[0]], sp[row[1]], ast[row[2]], tpr[row[3]]])
        Y.append(1 if row[4] == "Yes" else 2)

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    # Appended in for loop above

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0: #skipping the header
                    dbTest.append (row)

        TP, TN, FP, FN = 0, 0, 0, 0
        for data in dbTest: # data is essentially row
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            class_predicted = clf.predict([[age[data[0]], sp[data[1]], ast[data[2]], tpr[data[3]]]])[0]
            print(class_predicted)
            # print(class_predicted)
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if data[4] == "Yes" and class_predicted == 1:
                TP += 1
            if data[4] == "Yes" and class_predicted == 2:
                FN += 1
            if data[4] == "No" and class_predicted == 1:
                FP += 1
            if data[4] == "No" and class_predicted == 2:
                TN += 1
        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        min_acc = 1
        acc = (TP + TN) / (TP + TN + FP + FN)
        # print(acc)
        if acc < min_acc:
            min_acc = acc

    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("Final accuracy when training on ", ds, ": ", acc, sep="")