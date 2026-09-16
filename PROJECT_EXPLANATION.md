# Project Explanation: Loan Approval Prediction

This document explains the machine-learning project in simple language. The main implementation is in `notebooks/02_data_preparation_and_model.ipynb`.

## 1. Problem definition

We want to predict `Loan_Status`:

- `Y`: loan approved
- `N`: loan rejected

This is a **binary classification** problem because there are two possible classes.

The dataset contains applicant information such as income, credit history,
education, employment status, and property area. `Loan_ID` identifies an
application, but it is not a meaningful applicant characteristic, so it must
not be used as a model feature.

## 2. Load the data

```python
import pandas as pd
from pathlib import Path

project_dir = Path.cwd()
if project_dir.name == "notebooks":
    project_dir = project_dir.parent

data_path = project_dir / "data" / "raw.csv"
df = pd.read_csv(data_path)
```

`pandas` provides the DataFrame table structure. `Path` helps construct file
paths safely. The `project_dir` logic supports running the notebook from either
the project folder or the `notebooks` folder. `read_csv()` loads the CSV into
`df`.

Useful first checks are:

```python
print(df.shape)
print(df.columns.tolist())
display(df.head())
```

`shape` reports rows and columns. `columns` reports the available fields.
`head()` displays the first five records so we can check that loading worked.

## 3. Inspect data quality

```python
display(df.dtypes.to_frame("dtype"))
display(df.isna().sum().to_frame("missing_values"))
print("Duplicate rows:", df.duplicated().sum())
display(df["Loan_Status"].value_counts(dropna=False))
```

These checks answer four basic questions:

- What type is each column?
- Which columns contain missing values?
- Are complete rows duplicated?
- Does the target contain the expected values?

Missing values are expected in this dataset. For example, `Credit_History`,
`Self_Employed`, and `Loan_Amount_Term` may contain blanks. We should not
silently delete every incomplete row because this dataset is small.

Additional checks that are useful during exploration include:

```python
print(df.nunique())
print(df["Loan_ID"].is_unique)
print(df.columns.str.strip().tolist())
print(df["Loan_Status"].astype("string").str.strip().unique())
```

These checks look for constant columns, duplicate identifiers, accidental
spaces in column names, and unexpected target labels.

## 4. Clean simple representation problems

`Dependents` can contain the text value `3+`. It represents three or more
dependents, so we convert it to the numeric value `3`:

```python
df_model = df.copy()
df_model["Dependents"] = (
    df_model["Dependents"]
    .replace("3+", "3")
)
df_model["Dependents"] = pd.to_numeric(
    df_model["Dependents"],
    errors="coerce"
)
```

`errors="coerce"` converts values that cannot be interpreted as numbers into
missing values. The preprocessing pipeline will handle those missing values
later.

The target is validated and encoded as follows:

```python
valid_statuses = {"Y", "N"}
invalid_statuses = set(df_model["Loan_Status"].dropna()) - valid_statuses
if invalid_statuses:
    raise ValueError(f"Unexpected Loan_Status values: {invalid_statuses}")

df_model["Loan_Status"] = df_model["Loan_Status"].map({
    "Y": 1,
    "N": 0
})
```

The model needs numbers, so approved applications become `1` and rejected
applications become `0`. Validation happens first so an unexpected label does
not silently become a missing target.

## 5. Create useful features

```python
df_model["TotalIncome"] = (
    df_model["ApplicantIncome"] +
    df_model["CoapplicantIncome"]
)

df_model["LoanToIncomeRatio"] = (
    df_model["LoanAmount"] /
    df_model["TotalIncome"].replace(0, pd.NA)
)

df_model["HasCoapplicant"] = (
    df_model["CoapplicantIncome"] > 0
).astype(int)
```

`TotalIncome` combines the applicant and co-applicant income. The loan-to-
income ratio provides a simple measure of loan size relative to income. The
zero-income replacement prevents division by zero. `HasCoapplicant` records
whether co-applicant income is present.

These features are acceptable only if the information would be available when
a loan application is assessed. Using future repayment information or a
post-approval decision would be data leakage.

## 6. Separate features and target

```python
X = df_model.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
y = df_model["Loan_Status"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing target values:", y.isna().sum())
```

`X` contains the input columns. `y` contains the answer the model is trying
to learn. `Loan_ID` is removed because an identifier can cause memorization
without representing a real relationship.

The target must have no missing values after mapping. If it does, inspect the
original labels before continuing.

## 7. Split the data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The training set is used to learn the model. The test set is held back for a
final unbiased check. `test_size=0.20` reserves 20 percent for testing.
`random_state=42` makes the split repeatable. `stratify=y` keeps the approved
and rejected proportions similar in both sets, which is important because the
classes are not equally frequent.

The test set should not be used to fit imputers, scalers, encoders, or models.
Otherwise, information from the evaluation data leaks into training.

## 8. Build the preprocessing pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["str", "category"]
).columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])
```

Numeric and categorical columns require different treatment:

- **Imputation** fills missing values. The numeric median is less affected by
  unusually large values than the mean. The most frequent category is a simple
  baseline for categorical data.
- **Scaling** puts numeric columns on comparable ranges. This is especially
  useful for Logistic Regression.
- **One-hot encoding** changes categories such as `Urban` and `Rural` into
  numeric indicator columns.
- `handle_unknown="ignore"` prevents an error if a future applicant contains a
  category that was not present during training.
- `ColumnTransformer` applies the numeric and categorical pipelines to the
  correct columns.

The pipeline is fitted only when the model is fitted on `X_train`. Therefore,
statistics such as medians, category lists, and scaling values are not learned
from `X_test`.

## 9. Train baseline models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}
```

`class_weight="balanced"` gives more influence to the less frequent class.
It can help when mistakes on rejected applications are important, but it does
not automatically solve class imbalance.

The notebook combines each model with the same preprocessing transformer:

```python
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", models["Logistic Regression"])
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
```

Logistic Regression learns a weighted linear relationship and is relatively
easy to explain. A Decision Tree makes a sequence of if/then splits. A Random
Forest combines many trees and can model more complex patterns.

## 10. Evaluate the models

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
```

The metrics mean:

- **Accuracy**: the proportion of all predictions that were correct.
- **Precision**: among predicted approvals, how many were actually approved.
- **Recall**: among actual approvals, how many were found by the model.
- **F1-score**: a balance between precision and recall.
- **ROC-AUC**: how well the probability scores rank approved applications above
  rejected applications across many thresholds.

A confusion matrix contains:

- **True positive**: correctly predicted approval.
- **True negative**: correctly predicted rejection.
- **False positive**: rejected application predicted as approved.
- **False negative**: approved application predicted as rejected.

For this project, the important metric depends on the bank's objective. A
false positive can increase credit risk. A false negative can deny a reliable
applicant. Accuracy alone may hide poor performance on the smaller class.

The current educational comparison selected Logistic Regression as the best
baseline on the available split. That does not make it suitable for real
banking decisions; the dataset is small and a different split may produce
different results.

## 11. Threshold analysis

```python
y_probability = model_pipeline.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.50).astype(int)
```

`predict_proba()` returns probabilities. The default threshold is usually
`0.50`, but it is not automatically the correct business threshold. Increasing
the threshold can make approval more conservative, often reducing false
approvals while increasing false rejections.

Threshold selection should be based on documented business costs, fairness
checks, and validation data rather than chosen from the test set alone.

## 12. Cross-validation

A single split can be unstable when there are only a few hundred rows. A
stratified cross-validation estimate is more dependable:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    model_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

for metric_name in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
    print(metric_name, scores[metric_name].mean())
```

Cross-validation splits the training data several ways. Each validation fold
is used once for checking while the other folds are used for fitting. The test
set remains untouched until the final evaluation.

## 13. Interpret features carefully

For Logistic Regression, coefficients describe the direction and strength of
the model's association with the transformed features. A positive coefficient
increases the model's score for approval; a negative coefficient decreases it.
These are model associations, not proof of causation.

For tree models, `feature_importances_` shows how much the trained tree splits
used each transformed feature. Importance can be biased by feature structure
and should not automatically be interpreted as causal evidence.

## 14. Predict a new applicant

A new applicant must use the same original feature names and the same feature
engineering steps used during training:

```python
new_applicant = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": 0.0,
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000.0,
    "CoapplicantIncome": 1500.0,
    "LoanAmount": 120.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}])

new_applicant["TotalIncome"] = (
    new_applicant["ApplicantIncome"] +
    new_applicant["CoapplicantIncome"]
)
new_applicant["LoanToIncomeRatio"] = (
    new_applicant["LoanAmount"] /
    new_applicant["TotalIncome"]
)
new_applicant["HasCoapplicant"] = (
    new_applicant["CoapplicantIncome"] > 0
).astype(int)

new_applicant = new_applicant[X.columns]
probability = model_pipeline.predict_proba(new_applicant)[0, 1]
prediction = int(probability >= 0.50)

print(f"Approval probability: {probability:.2%}")
print("Approved" if prediction == 1 else "Rejected")
```

The final column order matters because the trained pipeline expects the same
feature structure. In the project app, the saved metadata stores this expected
column list and the selected decision threshold.

## 15. Final checklist

Before considering the project complete, confirm:

- The target column is exactly `Loan_Status`.
- Only `Y` and `N` target values are used.
- `Loan_ID` is removed before modeling.
- Missing values are handled inside the pipeline.
- Categorical values are one-hot encoded.
- The preprocessing is fitted only on training data.
- The test set is not used during training or tuning.
- Evaluation includes more than accuracy.
- The selected model and selection criterion are documented.
- Feature relationships are not described as causal.
- The small dataset limitation is clearly stated.
- The notebook runs from beginning to end.
