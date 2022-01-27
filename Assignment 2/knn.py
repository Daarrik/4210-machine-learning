#-------------------------------------------------------------------------
# AUTHOR: Darrik Houck
# FILENAME: knn.py
# SPECIFICATION: Calculates LOO-CV error rate for 1NN of binary_points.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

errors = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for j, compare_instance in enumerate(db):
        if i != j:
            X.append([int(compare_instance[0]), int(compare_instance[1])])
            Y.append(1 if compare_instance[-1] == "+" else 2)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    # Y = []
    # for j, compare_class in enumerate(db):
    #     if i != j:
    #         Y.append(1 if compare_class[-1] == "+" else 2)
    # Moved to a single for loop above

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [int(instance[0]), int(instance[1])]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]
    # print(class_predicted)

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != (1 if instance[-1] == "+" else 2):
        errors += 1

#print the error rate
#--> add your Python code here
print("Error rate:", errors/len(db))
