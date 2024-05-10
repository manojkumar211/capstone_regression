from data_wrangling import df
import statsmodels.formula.api as smf

X=df.iloc[:,[4,5,6]]
y=df['sales']

smf_model=smf.ols('y~X',data=df).fit()
print(smf_model.summary())