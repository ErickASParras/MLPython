from KNeigborsClassUE import KNeigborsClassUE
from NBayesClassUE import NBayesClassUE
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import pandas as pd

iris = datasets.load_iris()
X,Y = iris.data, iris.target
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.5,random_state=1234)

# KNN data
data0=pd.read_csv('dadosnumericos/iris.csv',header=None,skiprows=1)
data0X = np.array(data0.iloc[:,:-1].values.tolist())
data0Y = np.array(data0.iloc[:,-1:].values.tolist())
X_traindf0, X_testdf0, y_traindf0, y_testdf0 = train_test_split(data0X,data0Y,test_size=0.5,random_state=1234)

data1=pd.read_csv('dadosnumericos/rice.csv',header=None,skiprows=1)
data1X = np.array(data1.iloc[:,:-1].values.tolist())
data1Y = np.array(data1.iloc[:,-1:].values.tolist())
X_traindf1, X_testdf1, y_traindf1, y_testdf1 = train_test_split(data1X,data1Y,test_size=0.5,random_state=1234)


data2=pd.read_csv('dadosnumericos/wdbc.csv',header=None,skiprows=1)
data2X = np.array(data2.iloc[:,:-1].values.tolist())
data2Y = np.array(data2.iloc[:,-1:].values.tolist())
X_traindf2, X_testdf2, y_traindf2, y_testdf2 = train_test_split(data2X,data2Y,test_size=0.5,random_state=1234)

# Tests -> KNN
print("k = 1, p = 1 tests:\n")

Itest0 = KNeigborsClassUE(k=1,p=1)
Itest0.fit(X_traindf0,y_traindf0)
Itest0.predict(X_testdf0)
print("Iris score : ")
print(Itest0.score(y_testdf0))

# Rtest0 = KNeigborsClassUE(k=1,p=1)
# Rtest0.fit(X_traindf1,y_traindf1)
# Rtest0.predict(X_testdf1)
# print("Rice score : ")
# print(Rtest0.score(y_testdf1))

Wtest0 = KNeigborsClassUE(k=1,p=1)
Wtest0.fit(X_traindf2,y_traindf2)
Wtest0.predict(X_testdf2)
print("Wdbc score : ")
print(Wtest0.score(y_testdf2))


#####################################################################
# Nbayers data
data3=pd.read_csv('dadosnominais/bc-nominal.csv',header=None,skiprows=1)
data3X = np.array(data3.iloc[:,:-1].values.tolist())
data3Y = np.array(data3.iloc[:,-1:].values.tolist())
X_traindf3, X_testdf3, y_traindf3, y_testdf3 = train_test_split(data3X,data3Y,test_size=0.5,random_state=1234)

data4=pd.read_csv('dadosnominais/contact-lenses.csv',header=None,skiprows=1)
data4X = np.array(data4.iloc[:,:-1].values.tolist())
data4Y = np.array(data4.iloc[:,-1:].values.tolist())
X_traindf4, X_testdf4, y_traindf4, y_testdf4 = train_test_split(data4X,data4Y,test_size=0.5,random_state=1234)

data5=pd.read_csv('dadosnominais/weather-nominal.csv',header=None,skiprows=1)
print(data5.head())
data5X = np.array(data5.iloc[:,:-1].values.tolist())
data5Y = np.array(data5.iloc[:,-1:].values.tolist())
X_traindf5, X_testdf5, y_traindf5, y_testd5 = train_test_split(data5X,data5Y,test_size=0.5,random_state=1234)


meow = KNeigborsClassUE(k=3,p=1)
meow.fit(X_train,y_train)
predictions = meow.predict(X_test)
# print(X_train)
#print(predictions)
#print(predictions ==test0.predict(X_testdf0) )
# print(meow.score(y_test))
