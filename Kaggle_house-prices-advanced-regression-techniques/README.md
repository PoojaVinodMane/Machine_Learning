# House Prices Prediction Using Gradient Boosting Regressor

This project predicts house prices by preprocessing the data, training a **Gradient Boosting Regressor**, and generating predictions. It evaluates the model's performance using **Root Mean Squared Error (RMSE)** and optimizes results by addressing issues like overfitting through **cross-validation**. 

## Introduction

The task involves training and evaluating a model to predict house prices based on a dataset with both numerical and categorical features. The target variable, `SalePrice`, is log-transformed to handle skewness and variance issues. Two models were tested:
- **Gradient Boosting Regressor**: Achieved an RMSE of **0.12707**, offering better performance.
- **Random Forest**: RMSE was higher compared to Gradient Boosting Regressor.

---

## Preprocessing

1. **Numeric Features**:
   - Imputed missing values using the median.
   - Scaled features to prevent bias caused by larger feature values.

2. **Categorical Features**:
   - Handled missing values by imputing a constant placeholder.
   - Encoded features using `LabelEncoder` to ensure consistency between training and testing data.

3. **Target Variable (`SalePrice`)**:
   - Log-transformed to stabilize variance and reduce skewness.

---

## Model Training

1. Data is split into training and validation sets.
2. **Gradient Boosting Regressor** is trained on the training set.
3. RMSE is computed on the validation set for performance evaluation.
4. The model is retrained on the entire dataset for final predictions.
5. Predictions are transformed back to the original scale and saved in a submission file.

---

## How to Run

To preprocess the data, train the model, and generate predictions, run the following command:

```bash
python3 house-prices-advanced-regression-techniques.py
