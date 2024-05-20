from data_wrangling import df
import statsmodels.formula.api as smf

X1=df.iloc[:,[4,5,6]]
y1=df['sales']

smf_model=smf.ols('y1~X1',data=df).fit()
print(smf_model.summary())


X=df.iloc[:,[4,5]]
y=df['sales']
smf_model1=smf.ols('y~X',data=df).fit()
print(smf_model1.summary())
