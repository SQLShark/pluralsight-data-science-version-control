# Product Service
from flask import Flask, request,jsonify
import numpy as np
import os.path
import pickle
from sklearn import linear_model

my_path = os.path.abspath(os.path.dirname(__file__))


SaleConditionMapping = {"Normal":1, "Abnorml":2, "Partial": 3, "AdjLand": 4, "Alloca": 5, "Family": 6}
BldgTypeMapping = {"1Fam":1, "2fmCon":2, "TwnhsE": 3, "TwnhsE": 4, "Twnhs": 5}

app = Flask(__name__)
#api = Api(app)

PickleModelPath = os.path.join(my_path, "HousePriceModel.pkl")

@app.route('/score', methods=['POST'])
def score():  

    req_data = request.get_json()

    YearBuilt = req_data['YearBuilt']
    GrLivArea = req_data['GrLivArea']
    Fireplaces = req_data['Fireplaces'] #two keys are needed because of the nested object
    SaleCondition = req_data['SaleCondition'] #an index is needed because of the array
    BldgType = req_data['BldgType']

    try:
        SaleConditionMappingResult = int(SaleConditionMapping[SaleCondition])
    except:
        SaleConditionMappingResult = 1
    
    try:
        BldgTypeMappingResult = int(BldgTypeMapping[BldgType])
    except: 
        BldgTypeMappingResult = 1

    Params = np.array([[YearBuilt, GrLivArea, Fireplaces, SaleConditionMappingResult, BldgTypeMappingResult]])
   
    with open(PickleModelPath, 'rb') as k:
        PickleModel = pickle.load(k)
    Answer = PickleModel.predict(Params)
    return jsonify(response=int(Answer))

@app.route('/train', methods=['GET','POST'])
def train():
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
    return jsonify(message="Model has been retrained. Run /score to score model")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5071, debug=True)

