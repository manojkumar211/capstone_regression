from joblib import load
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures


df_new=pd.DataFrame({'tv_power':[443251.046669,537460.169168],'radio_root':[14405.009866,4302.442699]})

poly=PolynomialFeatures(degree=3)
data=poly.fit_transform(df_new)

model_file='file_pickle.pickle'

final_model=pickle.load(open(model_file,'rb'))
print(final_model.predict(data))



