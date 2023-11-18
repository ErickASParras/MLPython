import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score


class NBayesClassUE:
    def __init__(self,alpha = 1) -> None:
        self.alpha = alpha
        
    def fit(self,X,Y):
        self.X_train = X
        self.uniques = np.unique(Y)
        # Making a temporary dictonary where all unique values have a increasing number from 0
        values = {value:number for number, value in enumerate(self.uniques)}
        # Changing Y so it has the numbers instead of names
        Yfinal = np.vectorize(values.get)(Y)
        #makes the 2d array into a 1d arrays like the target from sckitlearn
        self.Y_train = Yfinal.ravel()
        return
    
    def score(self,Y):
        # Verify is the predictions have been made using predict()
        if self.predictions == None :
            return -1
        uniques = np.unique(Y)
        # Making a temporary dictonary where all unique values have a increasing number from 0
        values = {value:number for number, value in enumerate(uniques)}
        # Changing Y so it has the numbers instead of names
        Y = np.vectorize(values.get)(Y)
        Y = Y.ravel()
        pointsright = np.sum(self.predictions == Y ) / len(Y)
        return pointsright

    def predict(self,X):
        
        predictions = [self.predictpriv(t0) for t0 in X]
        self.predictions = predictions
        return predictions
    

    def predictpriv(self,X):



        self.alpha+(X)/(np.unique(self.X_train)*self.alpha+len(self.Y_train))#ESSE Y PRECISA SER ALTERADO PARA CONTAR APENAS O Y USADO AGORA

        pass
    
