# Author: Will Concannon
# Date: 1/3/18
# Purpose: Train an Artificial Neural Network for the Kaggle Titanic Competition
''' 
Work Needed:
    Find a better way to deal with missing data
    Find a way to include more features
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
dataset = df_train.append(df_test, ignore_index=True, sort=True)

# Replace NaNs
dataset.Embarked.fillna('S' , inplace=True )
dataset.Age.fillna(30 , inplace=True )
dataset.Fare.fillna(33.30 , inplace=True )

X = dataset.iloc[:,[0,2,3,5,7,8,9]]
y = dataset.iloc[:, 10]



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X.iloc[:,1] = labelencoder_X_1.fit_transform(X.iloc[:,1])

labelencoder_X_5 = LabelEncoder()
X.iloc[:,5] = labelencoder_X_5.fit_transform(X.iloc[:, 5])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Split into sets
X_train = X[:891,:] 
y_train = y[:891]
X_test = X[891:,:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import keras
import keras
from keras.models import Sequential
from keras.layers import Dense # The hidden layers
from keras.layers import Dropout

# Initialize ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(5, kernel_initializer='uniform', activation='relu', input_shape=(9,))) # Output dim is based on nodes in input layer + output layer divided by 2

# Add second hidden layer
classifier.add(Dense(5, kernel_initializer='uniform', activation='relu'))

# Add output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN using stochastic gradient descent
classifier.compile('adam', 'binary_crossentropy', metrics=['accuracy']) # Loss function is determined because we're using a binary sigmoid in the output

# Fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size=1, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
output.to_csv('prediction-ann.csv', index=False)






