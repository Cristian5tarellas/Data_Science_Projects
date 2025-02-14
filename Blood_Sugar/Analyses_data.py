#!/usr/bin/env python3

#%% Importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#%%% Loading data 
data = pd.read_csv('data/diabetes.csv')

#%% Exploration
print(f"\n[+] Understanding the dataset:")

# Length of Data
lenght_data = data.shape[0]
columns_data = data.shape[1]
print(f"\n\t[·] The number of patient in the dataset is: {lenght_data} women")
print(f"\t[·] The number of features is: {columns_data-1}")

# Display the first few rows of the DataFrame
data.head()


# %% Determining the most significant factors

# Dataset
X = data.values[:,0:8]
y = data.values[:,-1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

# Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Get feature importance (absolute values of coefficients)
feature_importance = np.abs(model.coef_[0])

# Evaluation of the Model
score = model.score(X_test, y_test)

print(f"\n[+] The Accuracy of the LogisticRegression model is {round(score*100,2)} %")

#%% Objective 1:
name = ["Pregnancies", "Glucose", "B. Pressure", "S. Thickness", "Insulin", "BMI", "D. Pedigree", "Age"]
plt.bar(name,feature_importance)
plt.xticks(rotation=45)
plt.show()

#%%
# Train a Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Get feature importance scores
importances = rf.feature_importances_
score = rf.score(X_test, y_test)

print(f"\n[+] The Accuracy of the LogisticRegression model is {round(score*100,2)} %")

print("Feature Importance:", importances)
#%% Objective 1:

print(name)
plt.bar(name,importances)
plt.xticks(rotation=45)
plt.show()
# %%
