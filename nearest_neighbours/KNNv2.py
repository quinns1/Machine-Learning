
# -*- coding: utf-8 -*-
"""
Name: Shane Quinn
Student Number: R00144107
Email: shane.quinn1@mycit.ie
Course: MSc Artificial Intelligence
Module: Practical Machine learning
Date: 09/11/2020
"""


import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing




import numpy as np

def main():
        
    training_data = np.genfromtxt('trainingData.csv', delimiter = ',')      #Read in Training
    test_data = np.genfromtxt('testData.csv', delimiter=',')                #Read in Test Data
    X_train = training_data[:,:-1]
    y_train = training_data[:,-1] 
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1] 
    k=3 
    
    
    defaultknn = KNeighborsRegressor(n_neighbors=k)
    defaultknn.fit(X_train, y_train)   
    print("With no preprocessing \nR2 score = ", defaultknn.score(X_test, y_test))
    

    normalizer = preprocessing.Normalizer().fit(X_train)
    n_X_train = normalizer.transform(X_train)
    n_X_test = normalizer.transform(X_test)    
    nknn = KNeighborsRegressor(n_neighbors=k)
    nknn.fit(n_X_train, y_train)   
    print("\nAfter Normalization\nR2 score = ", nknn.score(n_X_test, y_test))
        
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    s_X_train = scaler.transform(X_train)
    s_X_test = scaler.transform(X_test)
    sknn = KNeighborsRegressor(n_neighbors=k)
    sknn.fit(s_X_train, y_train)
    print("\nAfter Standardisation\nR2 score = ", sknn.score(s_X_test, y_test))
    
    
    
    

if __name__=='__main__':
    main()

