# Data:-

```
- Dataset having with 200 rows and 4 columns ['TV', 'radio', 'newspaper', 'sales'].
- In this dataset, all columns are in the same data type which is nothing but all are int types.
- In this dataset, we are not having any null values.
- The TV column having hight correlation (0.7822) with Sales column compare with other columns.
- The Radio column having (0.5762) % correlation with Sales.

```

## Tv_Column:-

```
- The TV column having with -0.069 skewness. which is we consider as a normal distribution. Its a left side skewness. if we want to apply Transformation technique weneed to apply Exponential, Power or Boxcox transformation.
- TV and Sales having 78% of correlation.
```

## Radio_column:-

```
- The Radio column having with 0.094 skewness. which is we consider as a normal distribution.Its a right side skewness. if we want to apply Transformation technique weneed to apply Log, Root or Boxcox transformation.
- Radio and Sales having 85& of Correlation.

```

## Newspaper_column:-

```
- The Newspaper column having with 0.894 skewness. which is we consider as a normal distribution. if we want to apply Transformation technique weneed to apply Log, Root or Boxcox transformation.
- The Newspapere column having some Outliers as well.
- Newspaper and Sales ahving 23% of Correlation.
```


## EDA:-

```
- In EDA process, We find some outliers in Newspaper column and replace them with lower limit and upper limit value.
- We did not find any null values in any one of the columns.
```

## Data Cleaning:-
```
- Data Cleaning process, We replace the outliers with lower limit and upper limit value.
```

## Data Wrangling:-
```
- We find some skewness in all cloumns and applyed feature transformation technique to make the dasta into symmetrical distribution.
- For regression model we wont apply the feature scaling technique. With applying and without applying the feature scaling technique we will get same result for regression model.
```

## Feature Selection:-
```
- In feature selection, we apply the wrapping method to find out the relationship between all 3 independent features and dependent feature based on p-values as well as we will get the R2 and adj-R2 values by applying the wrapping method.
- In wrapping method we applyed the OLS method.
- After applying the wrapping method, we abserve the relationship between independent features and dependent feature. The newspaper column is not having the relationship with dependent features which means the p-value is 0.835, nothing but we reject the null hypothesis.
- We got R2 value is 0.888 and adj-R2 value is 0.886.
- By applying OLS method on 2 independent features, we got R2 value is 0.888 and adj-R2 value is 0.887.
```
