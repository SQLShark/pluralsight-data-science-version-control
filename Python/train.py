import pandas as pd
import numpy as np
import os.path
from sklearn import linear_model
import pickle

my_path = os.path.abspath(os.path.dirname(__file__))

SaleConditionMapping = {"Normal":1, "Abnorml":2, "Partial": 3, "AdjLand": 4, "Alloca": 5, "Family": 6}
BldgTypeMapping = {"1Fam":1, "2fmCon":2, "TwnhsE": 3, "TwnhsE": 4, "Twnhs": 5}

PickleModelPath = os.path.join(my_path, "HousePriceModel.pkl")

data = os.path.join(my_path, "data\\houseprices\\train.csv")
train_df = pd.read_csv(data)
combined = [train_df]
for dataset in combined:
    dataset['SaleConditionMapping'] = dataset['SaleCondition'].map(SaleConditionMapping)
    dataset['BldgTypeMapping'] = dataset['BldgType'].map(BldgTypeMapping)
    dataset['BldgTypeMapping'] = dataset['BldgTypeMapping'].fillna(0)

train_df1 = train_df[['YearBuilt', 'GrLivArea','Fireplaces','SaleConditionMapping','BldgTypeMapping']]
train_df2 = train_df[['SalePrice']]

regr = linear_model.LinearRegression()
X_train = train_df1
Y_train = train_df2
regr.fit(X_train, Y_train)
with open(PickleModelPath, 'wb') as f:
        pickle.dump(regr, f)  
print("Model has been retrained. Run /score to score model")