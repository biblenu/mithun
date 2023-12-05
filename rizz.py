import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeclassifier
from sklearn import metrics
data_set=pd.read_csv(r'C:\Users\mithu\Downloads\diabetes.xlsx')
fea_col=['pregnancies','glucose','bloodpressure','insulin','bmi','age']
x=data_set[fea_col]
y=data_set.outcome
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
algo=DecisionTreeclassifier()
algo=algo.fit(x_train,y_train)
y_pred=algo.predict(x_test)
print(x_test)
print(y_pred)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))

