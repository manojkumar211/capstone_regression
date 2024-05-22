from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import median_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures





X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)



# Linear Regression Model:-


class Linear_regression:

    linear_model =LinearRegression() # type: ignore
    linear_model.fit(X_train, y_train) # type: ignore
    lr_coe=linear_model.coef_ # type: ignore
    lr_int=linear_model.intercept_ # type: ignore
    y_tr_pred=linear_model.predict(X_train) # type: ignore
    y_te_pred=linear_model.predict(X_test) # type: ignore
    train_score=linear_model.score(X_train,y_train) # type: ignore  
    test_score=linear_model.score(X_test,y_test) # type: ignore
    cross_val=cross_val_score(linear_model,X,y,cv=5).mean() # type: ignore
    lr_tr_mae=median_absolute_error(y_train,y_tr_pred) # type: ignore
    lr_tr_mse=mean_squared_error(y_train,y_tr_pred) # type: ignore
    lr_tr_rmse=np.sqrt(mean_squared_error(y_train,y_tr_pred)) # type: ignore
    lr_te_mae=median_absolute_error(y_test,y_te_pred) # type: ignore
    lr_te_msr=mean_squared_error(y_test,y_te_pred) # type: ignore
    lr_te_rmse=np.sqrt(mean_squared_error(y_test,y_te_pred)) # type: ignore

    try:

        def __init__(self,linear_model,lr_coe,lr_int,y_tr_pred,y_te_pred,train_score,test_score,cross_val,lr_tr_mae,lr_tr_mse,lr_tr_rmse,lr_te_mae,lr_te_msr,lr_te_rmse):

            self.linear_model=linear_model
            self.lr_coe=lr_coe
            self.lr_int=lr_int
            self.y_tr_pred=y_tr_pred
            self.y_te_pred=y_te_pred
            self.train_score=train_score
            self.test_score=test_score
            self.cross_val=cross_val
            self.lr_tr_mae=lr_tr_mae
            self.lr_tr_mse=lr_tr_mse
            self.lr_tr_rmse=lr_tr_rmse
            self.lr_te_mae=lr_te_mae
            self.lr_te_msr=lr_te_msr
            self.lr_te_rmse=lr_te_rmse


        def linear_regression_model(self):
            return self.linear_model
        def linear_regression_coe(self):
            return self.lr_coe
        def linear_regression_int(self):
            return self.lr_int
        def linear_regression_y_tr_pred(self):
            return self.y_tr_pred
        def linear_regression_y_te_pred(self):
            return self.y_te_pred
        def linear_regression_train_score(self):
            return self.train_score
        def linear_regression_test_score(self):
            return self.test_score
        def linear_regression_cross_val(self):
            return self.cross_val
        def linear_regression_tr_mae(self):
            return self.lr_tr_mae
        def linear_regression_tr_mse(self):
            return self.lr_tr_mse
        def linear_regression_tr_rmse(self):
            return self.lr_tr_rmse
        def linear_regression_te_mae(self):
            return self.lr_te_mae
        def linear_regression_te_msr(self):
            return self.lr_te_msr
        def linear_regression_te_rmse(self):
            return self.lr_te_rmse
        
    except Exception as e:
        print(e)


          



# Polynomial Regression Model:-


class Polynomial_regression:

    poly=PolynomialFeatures(degree=3)
    X_train_poly=poly.fit_transform(X_train)
    X_test_poly=poly.fit_transform(X_test)
    lr_poly=LinearRegression()
    lr_poly.fit(X_train_poly,y_train)
    poly_train_pred=lr_poly.predict(X_train_poly)
    poly_test_pred=lr_poly.predict(X_test_poly)
    poly_train_score=lr_poly.score(X_train_poly,y_train)
    poly_test_score=lr_poly.score(X_test_poly,y_test)
    poly_cross_val_score=cross_val_score(lr_poly,X,y,cv=5).mean()
    poly_tr_mae=median_absolute_error(y_train,poly_train_pred)
    poly_tr_mse=mean_squared_error(y_train,poly_train_pred)
    poly_tr_rmsc=np.sqrt(mean_squared_error(y_train,poly_train_pred))
    poly_te_mae=median_absolute_error(y_test,poly_test_pred)
    poly_te_mse=mean_squared_error(y_test,poly_test_pred)
    poly_te_rsmc=np.sqrt(mean_squared_error(y_test,poly_test_pred))
    

    try:

        def __init__(self,poly,X_train_poly,X_test_poly,lr_poly,poly_train_pred,poly_test_pred,poly_train_score,poly_test_score,poly_cross_val_score,
                     poly_tr_mae,poly_tr_mse,poly_tr_rmsc,poly_te_mae,poly_te_mse,poly_te_rsmc):

            self.poly=poly
            self.X_train_poly=X_train_poly
            self.X_test_poly=X_test_poly
            self.lr_poly=lr_poly
            self.poly_train_pred=poly_train_pred
            self.poly_test_pred=poly_test_pred
            self.poly_train_score=poly_train_score
            self.poly_test_score=poly_test_score
            self.poly_cross_val_score=poly_cross_val_score
            self.poly_tr_mae=poly_tr_mae
            self.poly_tr_mse=poly_tr_mse
            self.poly_tr_rms=poly_tr_rmsc
            self.poly_te_mae=poly_te_mae
            self.poly_te_mse=poly_te_mse
            self.poly_te_rsmc=poly_te_rsmc
            

        def poly_regression(self):
            return self.poly
        def poly_X_train_poly(self):
            return self.X_train_poly
        def poly_X_test_poly(self):
            return self.X_test_poly
        def poly_lr_poly(self):
            return self.lr_poly
        def poly_train_pred_regression(self):
            return self.poly_train_pred
        def poly_test_pred_regression(self):
            return self.poly_test_pred
        def poly_train_score_regression(self):
            return self.poly_train_score
        def poly_test_score_regression(self):
            return self.poly_test_score
        def poly_cross_val_score_regression(self):
            return self.poly_cross_val_score
        def poly_train_mae_regression(self):
            return self.poly_tr_mae
        def poly_train_mse_regression(self):
            return self.poly_tr_mse
        def poly_train_rmse_regression(self):
            return self.poly_tr_rmsc
        def poly_test_mae_regression(self):
            return self.poly_te_mae
        def poly_test_mse_regression(self):
            return self.poly_te_mse
        def poly_test_rmse_regression(self):
            return self.poly_te_rsmc
        
    except Exception as e:
        print(e)

        
    
# Lasso Regression Method:-


class Lassocv_regression:

    lasso_cv=LassoCV(alphas=None,max_iter=1000,cv=5)
    lasso_cv.fit(X_train,y_train)
    alpha_lasso_cv=lasso_cv.alpha_

    try:

        def __init__(self,lasso_cv,alpha_lasso_cv):

            self.lasso_cv=lasso_cv
            self.alpha_lasso_cv=alpha_lasso_cv

        def lasso_cv_regression(self):
            return self.lasso_cv
        def lasso_cv_alpha(self):
            return self.alpha_lasso_cv
        
    except Exception as e:
        print(e)

class Lasso_regression(Lassocv_regression):

    lasso_model=Lasso(Lassocv_regression.alpha_lasso_cv) # type: ignore
    lasso_model.fit(X_train,y_train)
    lasso_train_pred=lasso_model.predict(X_train)
    lasso_test_pred=lasso_model.predict(X_test)
    lasso_train_score=lasso_model.score(X_train,y_train)
    lasso_test_score=lasso_model.score(X_test,y_test)
    lasso_cross_val_score=cross_val_score(lasso_model,X,y,cv=5).mean()
    lasso_tr_mae=median_absolute_error(y_train,lasso_train_pred)
    lasso_tr_mse=mean_squared_error(y_train,lasso_train_pred)
    lasso_tr_rmse=np.sqrt(mean_squared_error(y_train,lasso_train_pred))
    lasso_te_mae=median_absolute_error(y_test,lasso_test_pred)
    lasso_te_mse=mean_squared_error(y_test,lasso_test_pred)
    lasso_te_rmse=np.sqrt(mean_squared_error(y_test,lasso_test_pred))


    try:

        def __init__(self, lasso_cv, alpha_lasso_cv,lasso_model,lasso_train_pred,lasso_test_pred,lasso_train_score,lasso_test_score,lasso_cross_val_score,
                    lasso_tr_mae,lasso_tr_mse,lasso_tr_rmse,lasso_te_mae,lasso_te_mse,lasso_te_rmse):
            
            self.lasso_cv=lasso_cv
            self.alpha_lasso_cv=alpha_lasso_cv
            self.lasso_model=lasso_model
            self.lasso_train_pred=lasso_train_pred
            self.lasso_test_pred=lasso_test_pred
            self.lasso_train_score=lasso_train_score
            self.lasso_test_score=lasso_test_score
            self.lasso_cross_val_score=lasso_cross_val_score
            self.lasso_tr_mae=lasso_tr_mae
            self.lasso_tr_mse=lasso_tr_mse
            self.lasso_tr_rmse=lasso_tr_rmse
            self.lasso_te_mae=lasso_te_mae
            self.lasso_te_mse=lasso_te_mse
            self.lasso_te_rmse=lasso_te_rmse

        def lasso_cv_regression(self):
            return super().lasso_cv
        def lasso_cv_alpha(self):
            return super().alpha_lasso_cv
        def lasso_model_regression(self):
            return self.lasso_model
        def lasso_train_pred_regression(self):
            return self.lasso_train_pred
        def lasso_test_pred_regression(self):
            return self.lasso_test_pred
        def lasso_train_score_regression(self):
            return self.lasso_train_score
        def lasso_test_score_regression(self):
            return self.lasso_test_score
        def lasso_cross_val_score_regression(self):
            return self.lasso_cross_val_score
        def lasso_train_mae_regression(self):
            return self.lasso_tr_mae
        def lasso_train_mse_regression(self):
            return self.lasso_tr_mse
        def lasso_train_rmse_regression(self):
            return self.lasso_tr_rmse
        def lasso_test_mae_regression(self):
            return self.lasso_te_mae
        def lasso_test_mse_regression(self):
            return self.lasso_te_mse
        def lasso_test_rmse_regression(self):
            return self.lasso_te_rmse
        
    except Exception as e:
        print(e)

        



