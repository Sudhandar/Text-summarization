import pandas as pd

data = pd.read_csv('Reviews.csv')
columns = list(data.columns)
new_columns = []
for col in columns:
    new_columns.append(col.lower())
    
data.columns = new_columns

