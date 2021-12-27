
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

def main():
        
    training_data = np.genfromtxt('trainingData.csv', delimiter = ',')      #Read in Training and Test data
    test_data = np.genfromtxt('testData.csv', delimiter=',')                #Read in Test Data
    predictions=[]   
                                                        
    for t in test_data:
        predictions.append(predict(training_data, t[:12]))                  #Find and Save Predicted target values
        
    predictions = np.array(predictions)                                     
    r2 = calculate_r2(test_data[:,-1], predictions)                         
    
    print(r2)
    


def calculate_distances(td, qi):
    """
    Function that calculates the euclidean distance between single point in feature space (qi) and all other data points in feature space (td)
    Return distances in 1D NumPy array (d)

    Parameters
    ----------
    td : 2D NUMPY ARRAY
        TRAINING DATA.
    qi : 1D NUMPY ARRAY
        QUERY INSTANCE.

    Returns
    -------
    d : 1D NUMPY ARRAY
        DISTANCES 

    """
    
    td = td[:,:12]                                      #Remove Target Values
    d = np.sqrt(np.sum((td-qi)**2,axis=1))              #Euclidean Distance - Square root of the sum of the squared 
                                                        #difference between query instance and all training data features  
    return d


def predict(training_data, query_instance):
    """
    Function that predicts regression value given 1D Numpy Array (query_instance). Acceots 2D NumPy Array (training_data). Using Distance
    Weighted K-Nearest Neighbours (Regression). This is achieved by calculating the sum of the k closest neighbours target value multipied by
    it's corrosponding weight. This is divided by the sum of all weights.
    
    Each instance weight is calculated by finding the inverse distance squared.  
    
    Return predicted regression value

    Parameters
    ----------
    training_data : 2D NUMPY ARRAY
        TRAINING DATA.
    query_instance : 1D NUMPY ARRAY
        QUERY INSTANCE.

    Returns
    -------
    prv : INT
        PREDICTED REGRESSION VALUE.

    """

    k = 3                                                               #Set K here (hardcoded to 3 as specified in assignment)
    distances = calculate_distances(training_data, query_instance)      #Calculate all distances
    sort_indices = np.argsort(distances)                                #Find indexes of closest instances in the training data to query instance
 
    num = np.sum(training_data[:,-1][sort_indices[0:k]]*(1/distances[sort_indices[0:k]]**2))     #Numerator: Sum of ('wi' * f(xi)) -> Where wi is weight (discussed above) 
                                                                                                 # and f(xi) is instance 'i's target regression value
     
    den = np.sum(1/(distances[sort_indices[0:k]]**2))                   #Denominator: Sum of (1/wi)^2 -> inverse weight squared                                
    prv = num/den                                                       #Predicted regression value
    
    return prv
    
    
    
    
    
def calculate_r2(true_target, predict_target):
    """
    Calculate R-squared value, which is a performance metric for KNN 
    This is achieved by calculating 1 minus the sum of squared residuals divided by the total sum of squares (mean)
    The better the model is the lower the sum of squared residuals (numerator is)
    r-squared is generally between 0 and 1. The closer to 1 the better. A negative number means our prediction model performs worse than the mean

    Parameters
    ----------
    true_target : 1D NUMPY ARRAY
        TARGET VALUES.
    predict_target : 1D ARRAY
        PREDICTED VALUES.

    Returns
    -------
    r2 : INT
        R-SQUARED VALUE.

    """
    
    m = true_target.shape[0]                                            #Number of test target values
    y_b = np.sum(true_target)/m                                         #Y-Bar Sum of target values/number of values
    sum_sqr_residuals = np.sum((predict_target-true_target)**2)         #Numerator: Sum of Predicted Values - Target values, squared (SUM OF SQUARED RESIDUALS)
    sum_sqrs = np.sum((y_b-true_target)**2)                             #Denominator: The sum of YBar - each target value individuals, squared (SUM OF TOTAL SQUARES, mean)
    r2 = 1-sum_sqr_residuals/sum_sqrs                                   #R squared performance
    
    return r2
    


if __name__=='__main__':
    main()

