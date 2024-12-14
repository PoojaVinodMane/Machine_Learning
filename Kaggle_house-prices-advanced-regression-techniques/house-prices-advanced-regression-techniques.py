import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

#Loading the data
training_data = pd.read_csv('train.csv') 
testing_data = pd.read_csv('test.csv')

#Function for preprocessing 
def preprocess_data(training_data, testing_data):
    #Separating numeric and categorical columns
    numeric_features_col = training_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features_col = [col for col in numeric_features_col if col not in ['SalePrice', 'Id']]
    
    categorical_features_col = training_data.select_dtypes(include=['object']).columns.tolist()
    
    #Removing SalePrice from features
    if 'SalePrice' in numeric_features_col:
        numeric_features_col.remove('SalePrice')
    
    #Imputing median for missing data and scaling for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    training_data[numeric_features_col] = numeric_imputer.fit_transform(training_data[numeric_features_col])
    testing_data[numeric_features_col] = numeric_imputer.transform(testing_data[numeric_features_col])
    
    #Scaling numeric features
    scaler = StandardScaler()
    training_data[numeric_features_col] = scaler.fit_transform(training_data[numeric_features_col])
    testing_data[numeric_features_col] = scaler.transform(testing_data[numeric_features_col])
    
    #Imputing constant for missing data and encoding for categorical features
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    training_data[categorical_features_col] = categorical_imputer.fit_transform(training_data[categorical_features_col])
    testing_data[categorical_features_col] = categorical_imputer.transform(testing_data[categorical_features_col])
    
    #Converting 'missing' into a valid category for LabelEncoder by ensuring all columns have the same categories
    label_encoders = {}
    for col in categorical_features_col:
        le = LabelEncoder()
        #Combining train and test data to ensure the encoder sees all categories, also 'missing' ones
        combined_data = pd.concat([training_data[col], testing_data[col]], axis=0)
        le.fit(combined_data)
        training_data[col] = le.transform(training_data[col])
        testing_data[col] = le.transform(testing_data[col])
        #Storing encoders for possible inverse transformation
        label_encoders[col] = le  

    #Preparing x and y
    x = training_data.drop(['SalePrice', 'Id'], axis=1)
    # Log transform target
    y = np.log1p(training_data['SalePrice'])  
    x_test = testing_data.drop(['Id'], axis=1)
    
    return x, y, x_test

#Creating the GradientBoosting model 
def create_model():
    model = GradientBoostingRegressor(
        n_estimators=1000,  #Number of boosting stages
        learning_rate=0.02,  #Step size for each boosting stage
        max_depth=3,         #Maximum depth of individual trees
        subsample=0.9,       #Fraction of samples to be used for fitting
        max_features='sqrt',  #Maximum number of features to consider for each tree
        random_state=42
    )
    return model

#Preprocessing the data
x, y, x_test = preprocess_data(training_data, testing_data)

#Spliting data into training and validation sets (80% training, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#Model creation
model = create_model()

#Training the model on the training data
model.fit(x_train, y_train)

#Predicting on the validation set
y_val_pred = model.predict(x_val)

#Calculating RMSE for the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f'GradientBoosting RMSE: {rmse_val:.4f}')

#Final prediction on test data (using training set)
model.fit(x, y)  
y_test_pred = model.predict(x_test)

#Converting predictions back to original scale by inverse of log transformation
y_test_pred = np.expm1(y_test_pred)

#Creating submission file
submission = pd.DataFrame({
    'Id': testing_data['Id'],
    'SalePrice': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created")
