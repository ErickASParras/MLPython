import numpy as np
import pandas as pd
from math import *
from collections import Counter
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

class KNeigborsClassUE:

    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.predictions = None

    def fit(self,X,Y):
        self.X_train = X
        uniques = np.unique(Y)
        # Making a temporary dictonary where all unique values have a increasing number from 0
        values = {value:number for number, value in enumerate(uniques)}
        # Changing Y so it has the numbers instead of names
        Yfinal = np.vectorize(values.get)(Y)
        self.Y_train = Yfinal
    
    def predict(self,X):
        predictions = [self.predictpriv(t0) for t0 in X]
        self.predictions = predictions
        return predictions

    def score(self,Y):
        # Verify is the predictions have been made using predict()
        if self.predictions == None :
            return -1
        pointsright = np.sum(self.predictions == Y ) / len(Y)
        return pointsright
    
    def minkowski_distance(self,x, y, p_value):
        # pass the p_root function to calculate
        # all the value of vector parallelly 
        if p_value == 1:
            result = np.abs(np.sum(np.array(x)-np.array(y)))
        elif p_value == 2:    
            result = np.sqrt(np.sum(np.power(np.array(x)-np.array(y), 2)))
        return result
        
    
    def predictpriv(self,X):
        # Calculate using the formula as prescribed
        distance = [self.minkowski_distance(t0, X, self.p)for t0 in self.X_train]
        # Chossing only the k amount
        k_indices = np.argsort(distance)[:self.k]
        # print(k_indices)
        k_labels = [self.Y_train[i] for i in k_indices]

        # Counting each to choose the correct label
        return Counter(k_labels).most_common()[0][0]