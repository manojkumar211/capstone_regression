from data import data_des
from eda import tv_column,radio_column,newspaper_column
from data import df
from models import Linear_regression,Polynomial_regression,Lasso_regression,Lassocv_regression



print(data_des.colum) # type: ignore
print(data_des.describ) # type: ignore
print("**"*20)
print(tv_column.tv_skew) # type: ignore
print("**"*20)
print(radio_column.radio_des) # type: ignore
print("**"*20)
print(radio_column.radio_skew) # type: ignore
print("**"*20)
print(newspaper_column.newspaper_skew) # type: ignore
print("**"*20)
print(df['newspaper'].skew()) # type: ignore

print(round(4.615335331740764e-06))

print('linear model :',Linear_regression.linear_model) # type: ignore
print("**"*20)
print('linear train score :',Linear_regression.train_score) # type: ignore
print("**"*20)
print('linear test score :',Linear_regression.test_score) # type: ignore
print("**"*20)
print('linear Cross val :',Linear_regression.cross_val) # type: ignore
print("**"*20)
print('MAE Train :',Linear_regression.lr_tr_mae) # type: ignore
print("**"*20)
print('MSE Train :',Linear_regression.lr_tr_mse) # type: ignore
print("**"*20)
print('RMSE Train :',Linear_regression.lr_tr_rmse) # type: ignore
print("**"*20)
print('MAE Test :',Linear_regression.lr_te_mae) # type: ignore
print("**"*20)
print('MSE Test :',Linear_regression.lr_te_msr) # type: ignore
print("**"*20)
print('RMSE Test :',Linear_regression.lr_te_rmse) # type: ignore
print("**"*20)
print('Polynomial Training Score :',Polynomial_regression.poly_train_score) # type: ignore
print("**"*20)
print('Polynomial Test Score :',Polynomial_regression.poly_test_score) # type: ignore
print("**"*20)
print('Lasso CV value :',Lassocv_regression.alpha_lasso_cv) # type: ignore
print("**"*20)
print('Lasso Train Score :',Lasso_regression.lasso_train_score) # type: ignore
print("**"*20)
print('Lasso Test Score :',Lasso_regression.lasso_test_score)   # type: ignore
print("**"*20)
print('Lasso alpha value :',Lasso_regression.alpha_lasso_cv) # type: ignore