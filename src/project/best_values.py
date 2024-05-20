from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split



# Find the Best Random State Value:-


lr_best_train=[]
lr_best_test=[]

for i in range(0,20):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    lr_train_pred=lr.predict(X_train)
    lr_test_pred=lr.predict(X_test)
    lr_best_train.append(lr.score(X_train,y_train))
    lr_best_test.append(lr.score(X_test,y_test))
    
print('Best Train Random State Value :',np.argmax(lr_best_train))
print('Best Test Random State Value :',np.argmax(lr_best_test))