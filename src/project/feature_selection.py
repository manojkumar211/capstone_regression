from data_wrangling import df
import statsmodels.formula.api as smf
import pandas as pd

X1=df.iloc[:,[4,5,6]]
y1=df['sales']

smf_model=smf.ols('y1~X1',data=df).fit()
print(smf_model.summary())


X=df.iloc[:,[4,5]]
y=df['sales']
smf_model1=smf.ols('y~X',data=df).fit()
print(smf_model1.summary())


from statsmodels.stats.outliers_influence import variance_inflation_factor

class vif_model:

    vif=pd.DataFrame()

    vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    vif['features']=X.columns

    def __init__(self,VIF,features):

        self.VIF=VIF
        self.features=features

    def vif_values(self):
        return self.VIF
    def vif_features(self):
        return self.features
    
print(vif_model.vif)


