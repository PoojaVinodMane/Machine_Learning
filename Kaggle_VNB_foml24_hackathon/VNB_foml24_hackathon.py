#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

#Reading training and test datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#separating the features and the target
x_train = df_train.drop(columns=["UID", "Target"])
y_train = df_train["Target"]

#encoding the target variable 
labels = LabelEncoder()
y = labels.fit_transform(y_train)

#splitting data into training and validation sets 
x_training, x_val, y_training, y_val = train_test_split(x_train, y, test_size=0.2, random_state=42)

#drop all columns if all the entries are zero
x_train = x_train.dropna(axis=1, how='all')

#filling missing values by replacing with mean value 
mean_values = x_training.mean(axis=0)
x_training_new = x_training.fillna(mean_values)
x_val_new = x_val.fillna(mean_values)

#Standardizing the data using mean and standard deviation
training_mean = x_training_new.mean(axis=0)
training_std = x_training_new.std(axis=0)

#Set features with zero standard deviation to avoid division by zero
training_std[training_std == 0] = 1

#Standardizing each set 
x_training_standard = (x_training_new - training_mean) / training_std
x_val_standard = (x_val_new - training_mean) / training_std

#Converting the standardized numpy arrays back to pandas DataFrames
x_training_standard_df = pd.DataFrame(x_training_standard, columns=x_training.columns)
x_val_standard_df = pd.DataFrame(x_val_standard, columns=x_val.columns)

#training random forest classifier 
model = RandomForestClassifier(
    n_estimators=90, #The number of trees in the forest.
    max_depth=8,#maximum depth
    max_features='sqrt',#number of feautures
    class_weight='balanced',#adjust weights to handle class imbalance
    random_state=42
)

#model trained using fit 
model.fit(x_training_standard_df, y_training)

#prediction on the validation set on basis of standarized data
y_val_predict = model.predict(x_val_standard_df)

#calculating F1 score on the validation set 
f1 = f1_score(y_val, y_val_predict, average='macro')
print("Macro F1 Score:", f1)

#predition function on test data
def predictions(test_file, predictions_file):
    #loading the test data
    test_df = pd.read_csv(test_file)
    x_test = test_df.drop(columns=["UID"])

    #filling missing values by replacing with mean value 
    x_test_new = x_test.fillna(mean_values)

    #checking for NaN values in test data after replacing
    if np.any(np.isnan(x_test_new)):
        print("Test data contains NaN values")

    #converting test data back to main daatframe and put in order columns with training data
    x_test_new_df = pd.DataFrame(x_test_new, columns=x_train.columns)

    #standarizing the test data using same mean and standard of training data
    x_test_standard = (x_test_new_df - training_mean) / training_std

    #Checking if NaN exists in standardized test data
    if np.any(np.isnan(x_test_standard)):
        print("Test data contains NaN after standardization")

    #making predictions on the test data
    y_test_predict = model.predict(x_test_standard)
    
    #Converting the predictions back to original labels
    predictions_labels = labels.inverse_transform(y_test_predict)

    #creating a submission dataframe file
    submission = pd.DataFrame({
        "UID": test_df["UID"],
        "Target": predictions_labels
    })

    #Saving the predictions to CSV file
    submission.to_csv(predictions_file, index=False)
    print("Submission file: " + predictions_file)


#submission file
predictions("test.csv", "cs22btech11035_submission_code.csv")


# In[17]:


import argparse

def fill_na_values(df, features, mean_values):
    df[features] = df[features].fillna(mean_values)
    
def make_predictions(test_fname, predictions_fname):
#TODO: complete this function to save predictions to the csv file predictions_fname
#this is an example, you need to modify the code below to fit your workflow
#### start code ####
    test = pd.read_csv(test_fname)
    features = [col for col in test.columns if col != "UID"]
    fill_na_values(test, features, mean_values)
   
    x_test = test.drop(columns=["UID"])

    fill_na_values(x_test, features, mean_values)

    x_test_standard = (x_test - training_mean) / training_std

    x_test_standard = pd.DataFrame(x_test_standard, columns=features)              

    test_X = x_test_standard[features].to_numpy()
    preds = model.predict(x_test_standard)                           
    test_uid = test[["UID"]].copy()
    test_uid["Target"] = labels.inverse_transform(preds)
    test_uid.to_csv(predictions_fname, index=False)
    print("Submission file: " + predictions_fname)
    
#### end code ####
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, help="file path of train.csv")
    parser.add_argument("--test-file", type=str, help="file path of test.csv")
    parser.add_argument("--predictions-file", type=str, help="save path of predictions")
    args = parser.parse_args()

    make_predictions(args.test_file, args.predictions_file)

