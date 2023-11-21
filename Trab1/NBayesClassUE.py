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
            print(self.Nclasses.get((str(i),)))
            print(list(self.dic.values()).count(i))
            self.priors.append(((self.Nclasses.get((str(i),))))/len(self.X_train))
        print(self.priors)
        # # Making a temporary dictonary where all unique values have a increasing number from 0
        # values = {value:number for number, value in enumerate(self.uniques)}
        # # Changing Y so it has the numbers instead of names
        # Yfinal = np.vectorize(values.get)(Y)
        # # Makes the 2d array into a 1d arrays like the target from sckitlearn
        # self.Y_train = Yfinal.ravel()
        self.mean ={}
        self.variance ={}
        for index, t0 in enumerate(self.uniques):            
                # print(t0)
                # print(self.dic[index])
                # Xt0 = self.dic[index] == t0
                # self.mean[index, :] = np.mean(Xt0,axis=0)
                # self.variance[index, :] = np.var(Xt0,axis=0)
                # self.priors[index] = np.array(Xt0).shape[0]
                u = 1
        
        # for i in self.dic.index(i):
        #     print(i)
        print(self.mean)
        print("-------------------\n")
        print(self.variance)
        print("-------------------\n")
        print(self.priors)
        print("-------------------\n")
        return
    
    def calculate_prior(self, Y):
        prior = []
        for i in self.uniques:
            count_i = np.sum(np.all(self.X_train == i, axis=1))
            prior.append(count_i / len(self.X_train))        
        print(prior)
        return prior

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
    
