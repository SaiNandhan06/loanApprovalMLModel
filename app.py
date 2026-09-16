from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "loan_approval_logistic_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "loan_approval_metadata.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata

model, metadata = load_artifacts()

st.title("🏦 Bank Loan Approval Predictor")
st.write(
    "Enter applicant information to receive a model-based loan approval prediction."
)

st.warning(
    "Educational project only. This tool is not a real banking decision system."
)

with st.form("loan_application_form"):
    st.subheader("Applicant Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=120.0,
        step=1.0,
        help="Use the same unit as your training dataset."
    )

    loan_amount_term = st.selectbox(
        "Loan Amount Term (months)",
        [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0],
        index=8
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda value: "Good / Available (1)" if value == 1.0 else "Poor / Not Available (0)"
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    total_income = applicant_income + coapplicant_income

    loan_to_income_ratio = (
        loan_amount / total_income
        if total_income > 0
        else 0
    )

    has_coapplicant = int(coapplicant_income > 0)

    applicant_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": float(dependents),
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "TotalIncome": total_income,
        "LoanToIncomeRatio": loan_to_income_ratio,
        "HasCoapplicant": has_coapplicant
    }])

    # Put input columns in exactly the same order used during training
    applicant_df = applicant_df[metadata["feature_columns"]]

    approval_probability = model.predict_proba(applicant_df)[0, 1]
    prediction = int(approval_probability >= metadata["threshold"])

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Approval Probability",
            f"{approval_probability:.2%}"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{metadata['threshold']:.2f}"
        )

    if prediction == 1:
        st.success("Predicted Decision: APPROVED")
    else:
        st.error("Predicted Decision: REJECTED")

    with st.expander("View model input"):
        st.dataframe(applicant_df, use_container_width=True)