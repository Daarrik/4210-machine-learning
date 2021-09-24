#-------------------------------------------------------------------------
# AUTHOR: Darrik Houck
# FILENAME: naive_bayes.py
# SPECIFICATION: Prints test cases from weather_test.csv which are over 0.75 confidence based on the data in weater_training.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: About 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

weather_training = []
weather_test = []

#reading the training data
#--> add your Python code here
with open("weather_training.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            weather_training.append (row)

outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temp = {"Cool": 1, "Mild": 2, "Hot": 3}
hum = {"Normal": 1, "High": 2}
wind = {"Weak": 1, "Strong": 2}
#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
Y = []
for row in weather_training:
    X.append([outlook[row[1]], temp[row[2]], hum[row[3]], wind[row[4]]])
    Y.append(1 if row[-1] == "Yes" else 2)

# for i, row in enumerate(weather_training):
#     print(weather_training[i])
#     print(X[i])
#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y = 

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
with open("weather_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            weather_test.append (row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
for row in weather_test:
    test_row = [outlook[row[1]], temp[row[2]], hum[row[3]], wind[row[4]]]
    predicted = clf.predict_proba([test_row])[0]
    # print(predicted)
    if predicted[0] > predicted[1] and predicted[0] >= 0.75:
        print(row[0].ljust(15), row[1].ljust(15), row[2].ljust(15), row[3].ljust(15), row[4].ljust(15), "Yes".ljust(15), "{:.3f}".format(predicted[0]), sep="")
    elif predicted[0] < predicted[1] and predicted[1] >= 0.75:
        print(row[0].ljust(15), row[1].ljust(15), row[2].ljust(15), row[3].ljust(15), row[4].ljust(15), "No".ljust(15), "{:.3f}".format(predicted[1]), sep="")
