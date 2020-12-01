import pandas as pd

# Show all column.
pd.set_option('display.max_columns', None)

pd.set_option('display.width', 1000)

df = pd.read_csv("./rossmann-store-sales/train.csv", low_memory=False)
print("Shape of the Dataset:",df.shape)

#the head method displays the first 5 rows of the data
print(df.head(5))

store = pd.read_csv("./rossmann-store-sales/store.csv")
print("Shape of the Dataset:",store.shape)
#Display the first 5 rows of data using the head method of pandas dataframe
print(store.head(5))

df_new = df.merge(store,on=["Store"], how="inner")
print(df_new.shape)

print("Distinct number of Stores :", len(df_new["Store"].unique()))
print("Distinct number of Days :", len(df_new["Date"].unique()))
print("Average daily sales of all stores : ",round(df_new["Sales"].mean(),2))

print(df_new.dtypes)

#We can extract all date properties from a datetime datatype
import numpy as np
df_new['Date'] = pd.to_datetime(df_new['Date'], infer_datetime_format=True)
df_new["Month"] = df_new["Date"].dt.month
df_new["Quarter"] = df_new["Date"].dt.quarter
df_new["Year"] = df_new["Date"].dt.year
df_new["Day"] = df_new["Date"].dt.day
df_new["Week"] = df_new["Date"].dt.isocalendar().week
df_new["Season"] = np.where(df_new["Month"].isin([3,4,5]),"Spring",np.where(df_new["Month"].isin([6,7,8]),"Summer",np.where(df_new["Month"].isin([9,10,11]),"Fall",np.where(df_new["Month"].isin([12,1,2]),"Winter","None"))))

#Using the head command to view (only) the data and the newly engineered features
print(df_new[["Date","Year","Month","Day","Week","Quarter","Season"]].head())