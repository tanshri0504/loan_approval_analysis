
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #2E86C1;
}
.sub-title {
    font-size: 18px;
    color: #5D6D7E;
}
.card {
    background-color: #F4F6F7;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">🏦 Loan Approval Analysis & Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Dashboard for Loan Eligibility Insights</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")

    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna('Yes', inplace=True)
    df['Dependents'].fillna(0, inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['LoanAmount'].fillna(128.0, inplace=True)
    df['Loan_Amount_Term'].fillna(360.0, inplace=True)
    df['Credit_History'].fillna(1.0, inplace=True)

    df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)

    return df

df = load_data()

# -----------------------------
# KPI METRICS
# -----------------------------
approved = df[df['Loan_Status'] == 'Y'].shape[0]
total = df.shape[0]
approval_rate = round((approved / total) * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("📄 Total Applications", total)
col2.metric("✅ Approved Loans", approved)
col3.metric("📊 Approval Rate", f"{approval_rate}%")

st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📌 Dashboard Menu")
option = st.sidebar.selectbox(
    "Choose Section",
    ["Dataset Overview", "EDA Visualizations", "Loan Approval Predictor"]
)

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
if option == "Dataset Overview":
    st.subheader("📄 Dataset Overview")

    with st.expander("🔍 View Raw Data"):
        st.dataframe(df)

    st.subheader("📌 Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("🧾 Missing Values")
    st.dataframe(df.isnull().sum().to_frame("Missing Count"))

# -----------------------------
# EDA VISUALIZATIONS
# -----------------------------
elif option == "EDA Visualizations":
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Loan Status vs Credit History**")
        fig, ax = plt.subplots()
        pd.crosstab(df['Loan_Status'], df['Credit_History']).plot(kind='bar', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Applicant Income Distribution**")
        fig, ax = plt.subplots()
        sb.boxplot(x=df['Loan_Status'], y=df['ApplicantIncome'], ax=ax)
        st.pyplot(fig)

    st.markdown("**Property Area vs Loan Status**")
    fig, ax = plt.subplots()
    pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

# -----------------------------
# LOAN APPROVAL PREDICTOR
# -----------------------------
elif option == "Loan Approval Predictor":
    st.subheader("🧮 Loan Approval Probability Checker")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("👤 Applicant Name")
        income = st.slider("💰 Applicant Income", 0, 30000, 5000)
        co_income = st.slider("🤝 Coapplicant Income", 0, 15000, 2000)
        loan_amt = st.slider("🏦 Loan Amount", 0, 600, 150)

    with col2:
        credit = st.radio("📊 Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Bad")
        education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
        married = st.selectbox("💍 Marital Status", ["Yes", "No"])

    if st.button("🔍 Check Loan Approval"):
        score = 0

        if credit == 1.0:
            score += 50
        if income > 5000:
            score += 20
        if co_income > 2000:
            score += 10
        if loan_amt < 200:
            score += 10
        if education == "Graduate":
            score += 5
        if married == "Yes":
            score += 5

        st.markdown("### 📊 Approval Probability")
        st.progress(score / 100)

        if score >= 70:
            st.success(f"✅ **{name}**, High Chance of Loan Approval ({score}%)")
        elif score >= 50:
            st.warning(f"⚠️ **{name}**, Moderate Chance of Loan Approval ({score}%)")
        else:
            st.error(f"❌ **{name}**, Low Chance of Loan Approval ({score}%)")

        st.info("📌 This prediction is based on rule-based logic for academic demonstration.")
