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
import seaborn as sns


# Importing the dataset
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
dataset = df_train.append(df_test, ignore_index=True, sort=True)

# Replace NaNs and Feature Engineering
x = lambda a : a['Name'].split(",")[1].split()[0]
dataset['Title'] = dataset.apply(x, axis=1)
Title_Dictionary = {
        "Capt.":       "Officer",
        "Col.":        "Officer",
        "Major.":      "Officer",
        "Dr.":         "Officer",
        "Rev.":        "Officer",
        "Jonkheer.":   "Royalty",
        "Don.":        "Royalty",
        "Sir." :       "Royalty",
        "the":"Royalty",
        "Dona.":       "Royalty",
        "Lady." :      "Royalty",
        "Mme.":        "Mrs",
        "Ms.":         "Mrs",
        "Mrs." :       "Mrs",
        "Mlle.":       "Miss",
        "Miss." :      "Miss",
        "Mr." :        "Mr",
        "Master." :    "Master"
}
dataset['Title'] = dataset.Title.map(Title_Dictionary)
dataset.loc[dataset.Age.isnull(), 'Age'] = dataset.groupby(['Sex','Pclass','Title']).Age.transform('median')
dataset.loc[dataset.Fare.isnull(), 'Fare'] = dataset.groupby(['Sex','Pclass','Title']).Fare.transform('median')
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])


X = dataset.iloc[:,[0,2,3,5,7,8,9,12]]
y = dataset.iloc[:, 10]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X.iloc[:,1] = labelencoder_X_1.fit_transform(X.iloc[:,1])

labelencoder_X_5 = LabelEncoder()
X.iloc[:,5] = labelencoder_X_5.fit_transform(X.iloc[:, 5])

labelencoder_X_7 = LabelEncoder()
X.iloc[:,7] = labelencoder_X_7.fit_transform(X.iloc[:, 7])

onehotencoder = OneHotEncoder(categorical_features = [1,7])
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
classifier.add(Dense(16, kernel_initializer='uniform', activation='relu', input_shape=(15,))) # Output dim is based on nodes in input layer + output layer divided by 2
classifier.add(Dropout(rate=0.6))

# Add second hidden layer
classifier.add(Dense(16, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.6))

# Add output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN using stochastic gradient descent
classifier.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy']) # Loss function is determined because we're using a binary sigmoid in the output

# Fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size=15, epochs=150)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
output.to_csv('prediction-ann.csv', index=False)



# Evaluate, improve, and tune the ANN

# Evaluate
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

# Wrap Keras functionality to use sklearn's K-Fold CV capabilities. 
def build_classifier(): # Needed for KerasClassifier
    classifier = Sequential()  
    classifier.add(Dense(70, kernel_initializer='uniform', activation='relu', input_shape=(15,))) 
    classifier.add(Dropout(rate=0.5))
    classifier.add(Dense(70, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.5))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=100, epochs=215)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5, n_jobs=1)

mean = accuracies.mean()
variance = accuracies.std()

# Tune

# Parameters to tune: # of Epoch, batch size, optimizer, number of neurons in layers
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

def build_classifier(optimizer, number_of_neurons, dropout_rate): # Needed for KerasClassifier
    classifier = Sequential()  
    classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu', input_shape=(15,))) 
    classifier.add(Dropout(rate=dropout_rate))
    classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=dropout_rate))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {
                'batch_size': sp_randint(1,100),
                'epochs':sp_randint(10,500),
                'optimizer':['adam','rmsprop'],
                'number_of_neurons': sp_randint(3,100),
                'dropout_rate': uniform(0.0,0.75)
              }

# Use Random Search instead of Grid Search
rs = RandomizedSearchCV(estimator=classifier, 
                        param_distributions=parameters,
                        n_iter=10,
                        scoring='accuracy',
                        cv=5,
                        verbose=1
                        )
rs.fit(X_train, y_train)

# Evaluate the grid search
best_parameters = rs.best_params_
best_accuracy = rs.best_score_



















