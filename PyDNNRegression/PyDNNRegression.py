import pandas as pd

# Show all column.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("./rossmann-store-sales/train.csv", low_memory=False)
print("Shape of the Dataset:",df.shape)

#the head method displays the first 5 rows of the data
print(df.head(5))