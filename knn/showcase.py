import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data Read
df = pd.read_csv('iris.data', delimiter=',', encoding="utf-8-sig")
print(df.head(10))

X = df[['sepal_length', ' sepal_width', ' petal_length', ' petal_width']]
y = df[' class']

#Spliting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)

#Model Deployment
from sklearn.neighbors import KNeighborsClassifier
k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)

#Accuracy Tests
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#User Input
print("Input each mesurement to predict the class")
usrinput = [[0, 0, 0, 0]]

usrinput[0][0] = input("Sepal Length (cm): ")
usrinput[0][1] = input("Sepal Width (cm): ")
usrinput[0][2] = input("Petal Length (cm): ")
usrinput[0][3] = input("Petal Width (cm): ")

res = neigh.predict(usrinput)

print("Predicted iris class: " + res[0])

#Converting str classes to int for plotting
typeofiris = np.empty(150, dtype='object')
for label in range(150):
    if y[label] == 'Iris-setosa':
        typeofiris[label] = 0
    if y[label] == 'Iris-versicolor':
        typeofiris[label] = 1
    if y[label] == 'Iris-virginica':
        typeofiris[label] = 2

#Plotting
plt.scatter(X[' petal_length'], X[' petal_width'], c=typeofiris, cmap=plt.cm.get_cmap('Set1', 3))
plt.xlabel('Pedal Length (cm)')
plt.ylabel('Petal Width (cm)')
cb = plt.colorbar()
classes = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
cb.set_ticks(typeofiris)
cb.set_ticklabels(y)
plt.show()