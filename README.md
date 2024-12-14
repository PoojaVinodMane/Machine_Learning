# Agricultural Field Efficiency Prediction

This project aims to predict the efficiency of agricultural fields, classifying them as **"high performing"**, **"moderately performing"**, or **"low performing"**. The model evaluates whether a field achieves its potential considering its resources, size, and other qualities. For example, a field may yield a decent output but could still be inefficient due to factors like market distance or improper resource allocation.

The solution utilizes a **Random Forest Classifier** as the ensemble method to process the VBN dataset and predict field efficiency. Data preprocessing includes handling missing values and standardizing the data. The performance of the model is measured using the **F1 score** on validation data.

## Dataset

The dataset for this project is provided as part of the [Kaggle VBN 2024 FOMML Hackathon](https://www.kaggle.com/competitions/vnb-foml-2024-hackathon). 

### Download Links:
- [train.csv](https://www.kaggle.com/competitions/vnb-foml-2024-hackathon)
- [test.csv](https://www.kaggle.com/competitions/vnb-foml-2024-hackathon)

## Code 
- The code make uses of Random forest classifier as an ensembler method for the VBN dataset.
- Preprocessing is done by handling missing values after standardizing the data.
- The F1 score is calculated at the end on the validation data.
  
## How to Run

To evaluate the model, use the following command:

```bash
bash eval.sh name_of_file
