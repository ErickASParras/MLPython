import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

class NBayesClassUE:
    def __init__(self,alpha = 1) -> None:
        self.alpha = alpha

        
    def fit(self,X,Y):  
        self.X_train = X
        self.Ytrain = Y
        self.length = len(X)
        # Making each row of X to become a key for their corresponding Y value
        self.dic = {tuple(Xrow): Yvalue for Xrow, Yvalue in zip(X, Y)}
        #self.dic = [{tuple(Xrow): tuple(Yvalue)} for Xrow, Yvalue in zip(X, Y)]        
        self.uniques = np.unique(Y)
        self.Nclasses = {}
        self.priors = []

        # Getting how much of each class there is
        for i in Y:
            i = tuple(i)
            if self.Nclasses.get(i):
                self.Nclasses[i] += 1
            else:
                self.Nclasses[i] = 1
        
        # Getting the prior probability for each class
        print(self.Nclasses)
        for i in self.uniques:
            # Transforming i into a string and into a tuple to match how Nclasses is stored
            self.priors.append((self.Nclasses.get((str(i),))+self.alpha)/len(self.X_train)+(self.alpha*self.length))
        print(self.priors)
        self.mean ={}
        self.variance ={}
        return

    def score(self,Y):
        # Verify is the predictions have been made using predict()
        if self.predictions == None :
            return -1
        # uniques = np.unique(Y)
        # # Making a temporary dictonary where all unique values have a increasing number from 0
        # values = {value:number for number, value in enumerate(uniques)}
        # # Changing Y so it has the numbers instead of names
        # Y = np.vectorize(values.get)(Y)
        # Y = Y.ravel()
        # print(Y)
        # print(self.predict)
        pointsright = np.sum(self.predictions == Y ) / len(Y)
        return pointsright
    
    def lidstone(self,x,y):
        for t0 in x:
            # print(t0)
            # print(t0 in next(iter(self.dic.keys())) )
            # # print(y)
             # print(t0)
            # print(self.dic.get(t0))
            # print(np.sum(self.dic.get(t0) == y))
            # print("lidstone")
            
            result = ((t0 in next(iter(self.dic.keys())))+self.alpha)/(self.Nclasses.get(y)+(self.alpha*len(x)))
        return result

    def predict(self,X):
        
        predictions = [self.predictpriv(t0) for t0 in X]
        self.predictions = predictions
        return predictions
    

    def predictpriv(self,X):

        results = []
        for idx, c in enumerate(self.Nclasses):
            prior = np.log(self.priors[idx])
            post = np.sum(np.log(self.lidstone(X,c)))
            t0 = prior+post
            results.append(t0)
        #self.alpha+(X)/(np.unique(self.X_train)*self.alpha+len(self.Y_train))#ESSE Y PRECISA SER ALTERADO PARA CONTAR APENAS O Y USADO AGORA
        print(results)
        finalidx=np.argmax(results)
        # print(list(self.Nclasses)[finalidx])
        return list(self.Nclasses.keys())[finalidx]


    
