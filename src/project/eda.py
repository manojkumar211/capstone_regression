import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import df


# Column TV

class tv_column:

    tv_type=df['TV'].dtype
    tv_des=df['TV'].describe()
    tv_nullvalue=df['TV'].isnull().sum()
    tv_shape=df['TV'].shape
    tv_skew=df['TV'].skew()
    tv_std=df['TV'].std(ddof=0)

    def __init__(self,tv_type,tv_des,tv_nullvalue,tv_shape,tv_skew,tv_std):

        self.tv_type=tv_type
        self.tv_des=tv_des
        self.tv_nullvalue=tv_nullvalue
        self.tv_shape=tv_shape
        self.tv_skew=tv_skew
        self.tv_std=tv_std

    def tv_column_type(self):
        return self.tv_type

    def tv_column_des(self):
        return self.tv_des

    def tv_column_nullvalue(self):
        return self.tv_nullvalue

    def tv_column_shape(self):
        return self.tv_shape

    def tv_column_skew(self):
        return self.tv_skew

    def tv_column_std(self):
        return self.tv_std
    
# Histo plot for TV column
    
fig,ax=plt.subplots(figsize=(10,5))
sns.histplot(data=df['TV'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/tv_histo.png")


# distribution plot for TV column

plt.plot(data=df['TV']) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/tv_dist.png")


# Boxplot plot for TV column

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['TV'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/tv_box.png")


    
# Column Radio


class radio_column:

    radio_dtype=df['radio'].dtype
    radio_des=df['radio'].describe()
    radio_null=df['radio'].isnull().sum()
    radio_skew=df['radio'].skew()
    radio_std=df['radio'].std(ddof=0)


    def __init__(self,radio_dtype,radio_des,radio_null,radio_skew,radio_std):

        self.radio_dtype=radio_dtype
        self.radio_des=radio_des
        self.radio_null=radio_null
        self.radio_skew=radio_skew
        self.radio_std=radio_std


    def radio_column_type(self):
        return self.radio_dtype
    
    def radio_column_des(self):
        return self.radio_des
    
    def radio_column_nullvalue(self):
        return self.radio_null
    
    def radio_column_skew(self):
        return self.radio_skew
    
    def radio_column_std(self):
        return self.radio_std
    


# Histo plot for Radio column
    
fig,ax=plt.subplots(figsize=(10,5))
sns.histplot(data=df['radio'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/radio_histo.png")


# distribution plot for Radio column

plt.plot(data=df['radio']) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/radio_dist.png")


# Boxplot plot for Radio column

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['radio'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/radio_box.png")



# Column newspaper


class newspaper_column:

    newspaper_dtype=df['newspaper'].dtype
    newspaper_des=df['newspaper'].describe()
    newspaper_null=df['newspaper'].isnull().sum()
    newspaper_skew=df['newspaper'].skew()
    newspaper_std=df['newspaper'].std(ddof=0)


    def __init__(self,newspaper_dtype,newspaper_des,newspaper_null,newspaper_skew,newspaper_std):

        self.newspaper_dtype=newspaper_dtype
        self.newspaper_des=newspaper_des
        self.newspaper_null=newspaper_null
        self.newspaper_skew=newspaper_skew
        self.newspaper_std=newspaper_std


    def newspaper_column_type(self):
        return self.newspaper_dtype
    
    def newspaper_column_des(self):
        return self.newspaper_des
    
    def newspaper_column_nullvalue(self):
        return self.newspaper_null
    
    def newspaper_column_skew(self):
        return self.newspaper_skew
    
    def newspaper_column_std(self):
        return self.newspaper_std
    


# Histo plot for newspaper column
    
fig,ax=plt.subplots(figsize=(10,5))
sns.histplot(data=df['newspaper'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/newspaper_histo.png")


# distribution plot for newspaper column

plt.plot(data=df['newspaper']) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/newspaper_dist.png")


# Boxplot plot for newspaper column

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['newspaper'],ax=ax) # type: ignore
plt.savefig("C:/Users/Archana Siripuram/Desktop/nareshit/plots/newspaper_box.png")



