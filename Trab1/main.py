from KNeigborsClassUE import KNeigborsClassUE
from NBayesClassUE import NBayesClassUE
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import pandas as pd

iris = datasets.load_iris()
X,Y = iris.data, iris.target
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.5,random_state=1234)

# plt.figure()
# plt.scatter(X[:,2],X[:,3], c=Y, cmap=cmap, edgecolor='k',s=20)
# plt.show()

# data=pd.read_csv('dadosnumericos/iris.csv',header=None,skiprows=1)
# print(data.head())

# data1=pd.read_csv('dadosnumericos/rice.csv',header=None,skiprows=1)
# print(data1.head())

# data2=pd.read_csv('dadosnumericos/wdbc.csv',header=None,skiprows=1)
# print(data2.head())




meow = KNeigborsClassUE(k=3,p=1)
meow.fit(X_train,y_train)
predictions = meow.predict(X_test)
print(predictions)
print(meow.score(y_test))
