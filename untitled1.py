#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
train = pd.read_csv('train.csv')

train.info()

null = train.isnull().sum()

#Droping new price column because it has maximum nan value and also Unnamed:0 because it is equivalent to index
train = train.drop(['Unnamed: 0', 'New_Price'], axis=1)

# Droping nan values
train = train.dropna()
train = train.reset_index(drop=True)


null = train.isnull().sum()

train['Location'].value_counts()
train['Fuel_Type'].value_counts()
train['Transmission'].value_counts()
train['Owner_Type'].value_counts()


# Lets split some columns to make a new feature
train_df = train.copy()
name = train_df['Name'].str.split(" ", n =2, expand = True)
train_df['Company'] = name[0]
train_df['Model'] = name[1]

train_df['Mileage'] = train_df['Mileage'].str.split(" ", n=1, expand = True).get(0)
train_df['Engine'] = train_df['Engine'].str.split(" ", n=1, expand = True).get(0)
train_df['Power'] = train_df['Power'].str.split(" ", n=1, expand = True).get(0)


train_df = train_df.drop(['Name'], axis = 1)
train_df['Mileage'] = train_df['Mileage'].astype(float)
train_df['Engine'] = train_df['Engine'].astype(int)
train_df.replace("null", np.nan, inplace = True)
train_df = train_df.dropna()
train_df = train_df.reset_index(drop=True)
train_df['Power'] = train_df['Power'].astype(float)

train_df['Company'].value_counts()
train_df['Company'] = train_df['Company'].replace('ISUZU', 'Isuzu')

#Handiling Rare Categorical Feature
cat_features = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']

for feature in cat_features:
    temp = train_df.groupby(feature)['Price'].count()/len(train_df)
    temp_df = temp[temp>0.01].index
    train_df[feature] = np.where(train_df[feature].isin(temp_df), train_df[feature], 'Rare')

train_df['Company'].value_counts()
train_df.info()
train_df['Seats'] = train_df['Seats'].astype(int)
#E.D.A.
features = [feature for feature in train_df.columns] 

import seaborn as sns
plt.scatter(x = "Year", y = "Price" , data = train_df )
sns.boxplot(x = 'Company' , y= 'Price', data= train_df)
for feature in features:
    df =train.copy()
    df[feature].hist(bins=20)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()

sns.distplot(train_df['Price'])

#as we can see price is not depends on location so we can drop this column
train_df = train_df.drop(['Location'], axis = 1)

#Encoding Categorical data
columns = ['Fuel_Type','Transmission','Owner_Type','Company','Model']
def categorical_ohe(multicolumns):
    df = train_df.copy()
    i = 0
    for feilds in multicolumns:
        print(feilds)
        d1 = pd.get_dummies(train_df[feilds])
        train_df.drop([feilds], axis = 1)
        if i == 0:
            df = d1.copy()
        else:
            df = pd.concat([df, d1], axis =1)
        i =i +1
    df = pd.concat([df,train_df], axis =1)
    return df

final_df = categorical_ohe(columns)
final_df = final_df.loc[:,~final_df.columns.duplicated()]


import datetime
now = datetime.datetime.now()
final_df['Year'] = final_df['Year'].apply(lambda x : now.year - x)

corr = final_df.corr()

#From correlation chart we can see that Kilometers_Driven and Seats are not impacting much on price prediction so we can drop them
data = final_df.drop(final_df[columns], axis =1 ) #Droping Categorical Columns
x = data.to_csv('data.csv', index = False)
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

#Fitting Model 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state= 0)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = 0)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train,y_train)
y_pred1 = xgb.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, svr_pred)
r2_score(y_test, y_pred)
r2_score(y_test, dt_pred)
r2_score(y_test, y_pred1)


def predict_price(Company,Model, Fuel_Type, Transmission,
                  Owner_Type, Year, Mileage, Engine, 
                  Power):
    
    com_index = np.where(X.columns==Company)[0][0]
    model_index = np.where(X.columns==Model)[0][0]
    fuel_index = np.where(X.columns==Fuel_Type)[0][0]
    trans_index = np.where(X.columns==Transmission)[0][0]
    owner_index = np.where(X.columns==Owner_Type)[0][0]
     
    x =np.zeros(len(X.columns))
    
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
    return rf.predict([x])[0]


predict_price('Hyundai','Creta','Diesel','Manual','First',5,19.67,1582,126.2)

#Export Model to pickel file
import pickle
with open('model.pickle','wb') as f:
    pickle.dump(rf,f)
# Export location and column information to a file that will be useful later on in our prediction application 
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))




    