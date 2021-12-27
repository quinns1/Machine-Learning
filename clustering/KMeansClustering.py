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
import random
import matplotlib.pyplot as plt

# random.seed(134561329879465555)




def main():
    
    train = np.genfromtxt('clustering_data.csv', delimiter = ',')           #File location here  
    x = []
    y = []    
    for a in range(1,10):
        x.append(a)
        y.append(restart_KMeans(train, a, 10, 10))    
    plt.plot(x,y)
    plt.ylabel("Distortion Cost")
    plt.xlabel("Centroids")
    plt.title("Elbow Plot")



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
    d = np.sqrt(np.sum((td-qi)**2,axis=1))              #Euclidean Distance
    
    return d


def generate_centroids(feature_data, k):
    """
    Picks 5 random features and assigns them as centroids

    Parameters
    ----------
    feature_data : 2D NUMPY ARRAY
        FEAUTURE DATA.
    k : INT
        NUMBE OF CENTROIDS.

    Returns
    -------
    centroids : 2D NUMPY ARRAY
        CENTROID LOCATIONS IN FEATURE SPACE.

    """
    
    centroids = [] 
    
    for i in range(k):
        centroids.append(random.choice(feature_data))
        
    centroids = np.array(centroids)
    return centroids


def assign_centroids(feature_data, centroids):
    """
    Function that finds and assigns the closest centroid to each feature to it.

    Parameters
    ----------
    feature_data : 2D NUMPY ARRAY
        FEATURE DATA.
    centroids : 2D NUMPY ARRAY
        CENTROID POSITIONS.

    Returns
    -------
    assigned_centroids : 1D NUMPY ARRAY
        INDEX OF CENTROID CORROSPONDING FEATURE IS ASSIGNED TO.

    """
    
    distances = []
    assigned_centroids = []

    for i in centroids:
        distances.append(calculate_distances(feature_data, i))          #Calculating distance from each point in feature space to each centroid        

    distances = np.array(distances)        
    mins = np.amin(distances, axis=0)                                   #Finding the minimum distance in each (closest centroid)
    i = 0 
    x = 0    
 
    while i < len(mins):                                                #Iterate through each feature vector
        while x < len(centroids):                                       #Iterate through each centroid
            if mins[i] == distances[x][i]:                              #If the the minimum distance is the same as the distance to this centroid assign centroid to feature
                assigned_centroids.append(x)                            
                x = 0
                break
            else:
                x += 1       
        i += 1     
        
    assigned_centroids = np.array(assigned_centroids)  
           
    return assigned_centroids
        

def move_centroids(feature_data, centroid_indices, current_centroids):
    """
    Function moves centroids by calculating the mean of each datapoint assigned to it and returns new centroid positions

    Parameters
    ----------
    feature_data : 2D NUMPY ARRAY
        FEATURE DATA.
    centroid_indices : 1D NUMPY ARRAY
        INDIXES OF ASSIGNING FEATURE TO CENTROID.
    current_centroids : 2D NUMPY ARRAY
        CENTROID LOCATIONS IN FEATURE SPACE.

    Returns
    -------
    new_centroids : 2D NUMPY ARRAY
        NEW CENTROID LOCATIONS IN FEATURE SPACE.

    """

    p = [0]*feature_data.shape[1]                   #Number of Features placeholder
    sums = [p]*len(current_centroids)               #Sums placeholder
    count = [p]*len(current_centroids)              #Count placeholder
    c = 0                                           #For each centroid 
    sums = np.array(sums)
    count = np.array(count)   

    while c < len(current_centroids):
        f = 0                                       #For each feature
        while f < feature_data.shape[1]:
            sums[c][f] = np.sum(feature_data[centroid_indices==c, f])       #Find some of each feauture assigned to each centroid
            count[c][f] = len(feature_data[centroid_indices==c, f])         #Find the number of features counted above. (for dividing later)             
            f += 1
        c += 1
    

    new_centroids = np.divide(sums,count)           #New Centroids = Mean posn of each feauture assigned to it
    
  
    return new_centroids


    
    
def calculate_cost(feature_data, centroid_indices, current_centroids):
    """
    Function that calculates distortion cost function which is the sum of the squared distance of each feature to it's relevant centroid

    Parameters
    ----------
    feature_data : 2D NUMPY ARRAY
        FEATURE DATA.
    centroid_indices : 1D NUMPY ARRAY
        CENTROID INDICES.
    current_centroids : 2D NUMPY ARRAY
        CURRENT CENTROID LOCATIONS IN FEATURE SPACE.

    Returns
    -------
    cost : INT
        DISTORTION COST.

    """
    
    cost = 0
    m = feature_data.shape[0]
    c = 0
    while c < len(current_centroids):        
        cost += np.sum((feature_data[centroid_indices==c]-current_centroids[c])**2)     #Squared sum of distance of all features to their closest centroid
        c += 1
   
    cost = cost/m                   #Mean - Divide by total number of instances in data set
    return cost


def restart_KMeans(feature_data, num_centroids, iterations, restarts):
    """
    Function which calls KMeans x numbe of times where x is the number of random restarts specified.

    Parameters
    ----------
    feature_data : 2D NUMPY ARRAY
        FEATURE DATA.
    num_centroids : INT
        NUMBER OF CENTROIDS (K).
    iterations : INT
        NUMBER OF ITERATIONS.
    restarts : INT
        NUMBER OF RANDOM RESTARTS.

    Returns
    -------
    best_cost : INT
        DISTORTION COST FUNCTION.
    """
    
    best_centroids = []
    first = True    
    
    for x in range(restarts):                                                       #Repeat 'restarts' number of times.
        cost=0
        for i in range(iterations):                                                 #Repeat 'iterations' number of times
            if cost==0:
                centroids = generate_centroids(feature_data, num_centroids)         #Generate random centroids
            assigned_centroids = assign_centroids(feature_data, centroids)          #Assign centroids to features
            centroids = move_centroids(feature_data, assigned_centroids, centroids) #Move centroids
            cost = calculate_cost(feature_data, assigned_centroids, centroids)      #Calculate distortion cost function

        if first == True:  
            best_centroids = centroids
            best_cost = cost
            first = False
        if cost<best_cost and cost != 0:
            best_centroids = centroids
            best_cost = cost

    return best_cost
     


if __name__=='__main__':
    
    main()









