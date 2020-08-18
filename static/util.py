import pickle
import json
import numpy as np
import pandas as pd

__company = None
__models = None
__fuel = None
__owner = None
__trans = None
__data_columns = None
__model = None

data = pd.read_csv('data.csv')
X= data.drop(['Price','Rare','Kilometers_Driven','Seats'], axis = 1)
y = data['Price']


#Splitting Dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def get_estimated_price(Company,Model, Fuel_Type, Transmission,
                  Owner_Type, Year, Mileage, Engine, 
                  Power):
    try:
        com_index = __data_columns.index(Company.lower())
    except:
        com_index = -1
        
    try:
        model_index = __data_columns.index(Model.lower())
    except:
        model_index = -1
    try:
        fuel_index = __data_columns.index(Fuel_Type.lower())
    except:
        fuel_index = -1
    try:
        trans_index = __data_columns.index(Transmission.lower())
    except:
        trans_index = -1
    try:
        owner_index = __data_columns.index(Owner_Type.lower())
    except:
        owner_index = -1
    x = np.zeros(len(__data_columns))
    x[53] = Year
    x[54] = Mileage
    x[55] = Engine
    x[56] = Power
    if com_index >= 0:
        x[com_index] = 1
        
    if model_index >= 0:
        x[model_index] = 1
    
    if fuel_index >= 0:
        x[fuel_index] = 1
    
    if trans_index >= 0:
        x[trans_index] = 1
    
    if owner_index >= 0:
        x[owner_index] = 1
    
    x = scaler.transform([x])[0]

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __company
    global __models
    global __fuel
    global __trans
    global __owner

    with open("./columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __company = __data_columns[7:22]
        __models = __data_columns[22:53]
        __fuel = __data_columns[:2]
        __trans = __data_columns[2:4]
        __owner = __data_columns[4:7]

    global __model
    if __model is None:
        with open('./model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_company_names():
    return __company
def get_models_names():
    return __models
def get_fuel_names():
    return __fuel
def get_trans_names():
    return __trans
def get_owner_names():
    return __owner
def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_company_names())
    print(get_models_names())
    print(get_fuel_names())
    print(get_trans_names())
    print(get_owner_names())
    print(get_estimated_price('Hyundai','Creta','Diesel','Manual','First',5,19.67,1582,126.2))
    print(get_estimated_price('Maruti','Ciaz','Petrol','Manual','First',2,21.56,1462,103.25))

