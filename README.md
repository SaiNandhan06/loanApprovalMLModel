# Bank Loan Approval Prediction

Beginner-friendly binary classification project using the Analytics Vidhya Loan
Prediction dataset from Kaggle.

Dataset: <https://www.kaggle.com/datasets/hossamhibrahem/loan-prediction-analytics-vidhya>

The model predicts whether a loan is approved (`Y`) or rejected (`N`). The
identifier column `Loan_ID` is removed before training.

## Project structure

```text
bank_loan_approval/
|-- data/raw.csv                         # Input dataset
|-- models/                              # Saved model and metadata
|-- notebooks/
|   |-- 01_eda.ipynb                     # Exploratory data analysis
|   `-- 02_data_preparation_and_model.ipynb
|-- app.py                               # Streamlit prediction app
|-- PROJECT_EXPLANATION.md               # Step-by-step teaching guide
|-- requirements.txt                     # Python dependencies
`-- README.md                            # Project overview and setup
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the downloaded CSV at `data/raw.csv`. Then open
`notebooks/02_data_preparation_and_model.ipynb` and run the cells from top to
bottom.

The notebook performs the following steps:

1. Loads and inspects the dataset.
2. Checks missing values, duplicates, target values, and data types.
3. Removes `Loan_ID` and converts `Dependents` values such as `3+`.
4. Creates `X` and binary `y`, where approved is `1` and rejected is `0`.
5. Splits the data using stratification.
6. Imputes missing values, scales numeric columns, and one-hot encodes categorical columns in a pipeline.
7. Trains Logistic Regression, Decision Tree, and Random Forest baselines.
8. Compares accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
9. Saves the selected Logistic Regression pipeline and prediction metadata.

## Run the Streamlit app

After running the notebook and creating the model artifacts:

```powershell
streamlit run app.py
```

The app accepts applicant details and displays an approval probability and
predicted decision. It is an educational project, not a real banking decision
system.

## Important limitations

This is a small educational dataset. A single train/test split can produce
unstable results, and the model must not be treated as reliable for real loan
decisions. Accuracy alone is insufficient: false approvals and false
rejections have different business costs. Fairness, calibration, threshold
selection, cross-validation, and monitoring would be required for a real
application.

For detailed explanations of the code, read
[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).