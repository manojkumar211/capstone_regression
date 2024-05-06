from data import data_des
from eda import tv_column,radio_column,newspaper_column



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