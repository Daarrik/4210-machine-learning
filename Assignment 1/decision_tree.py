#-------------------------------------------------------------------------
# AUTHOR: Darrik Houck
# FILENAME: decision_tree.py
# SPECIFICATION: Create a decision tree from contact_lens.csv
# FOR: CS 4200- Assignment #1
# TIME SPENT: 20 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
import os
os.chdir('C:/Users/Darrik/Documents/Python/4210/Assignment 1')
db = []
X = []
Y = []

#reading the data in a csv file
with open('asdf.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
# Age: Young = 1, Presbyopic = 2, Prepresbyopic = 3
# Spectacle Prescription: Myope = 1, Hypermetrope = 1
# Astigmatism: Yes = 1, No = 2
# Tear Production Rate: Normal = 1, Reduced = 2
# X should be:
X = [
      [1, 1, 1],
      [2, 3, 1],
      [2, 1, 2],
      [1, 3, 1],
      [2, 2, 1],
      [2, 3, 2],
      [1, 2, 1],
      [2, 1, 1]
    ]
# for row in db:
#     X_row = []
#     if row[0] == "Young":
#         X_row.append(1)
#     elif row[0] == "Presbyopic":
#         X_row.append(2)
#     elif row[0] == "Prepresbyopic":
#         X_row.append(3)

#     if row[1] == "Myope":
#         X_row.append(1)
#     elif row[1] == "Hypermetrope":
#         X_row.append(2)

#     if row[2] == "Yes":
#         X_row.append(1)
#     elif row[2] == "No":
#         X_row.append(2)

#     if row[3] == "Normal":
#         X_row.append(1)
#     elif row[3] == "Reduced":
#         X_row.append(2)

#     X.append(X_row)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Recommend Lenses: Yes = 1, No = 2
# for row in db:
#   if row[-1] == "Yes":
#     Y.append(1)
#   else:
#     Y.append(0)
Y = [0,0,0,0,1,0,0,1]

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=1)
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['home', 'marital', 'annual'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()