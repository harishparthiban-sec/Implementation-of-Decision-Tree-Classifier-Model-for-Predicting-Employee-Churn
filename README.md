# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: Pandas, Matplotlib, and DecisionTreeClassifier from scikit-learn.

2.Load the dataset and convert categorical data into numeric values using dummy variables.

3.Separate input features X and target variable Y.

4.Split the dataset into training and testing sets.

5.Create and train the Decision Tree Classifier model using the training data.

6.Predict test data results, calculate accuracy, and plot the decision tree diagram. 

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harish P
RegisterNumber:  212225040115
```

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(model,feature_names=x.columns,filled=True)

plt.show()
```

## Output:

<img width="1276" height="666" alt="image" src="https://github.com/user-attachments/assets/5aad677f-c71c-4230-886e-a11cda36a8ed" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
