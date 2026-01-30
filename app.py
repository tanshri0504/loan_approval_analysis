import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Approval Analysis", layout="wide")

st.title("🏦 Loan Approval Analysis & Prediction App")

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")
    return df

df = load_data()

st.subheader("Raw Dataset")
st.dataframe(df.head())

# -------------------- Data Cleaning --------------------
df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('Yes')
df['Dependents'] = df['Dependents'].fillna(0)
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(1.0)

df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)

st.success("Missing values handled successfully")

# -------------------- Exploratory Analysis --------------------
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.write("Loan Status Count")
    fig, ax = plt.subplots()
    sns.countplot(x=df['Loan_Status'], ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Credit History vs Loan Status")
    fig, ax = plt.subplots()
    pd.crosstab(df['Loan_Status'], df['Credit_History']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

st.write("Applicant Income Distribution")
fig, ax = plt.subplots()
sns.boxplot(x=df['Loan_Status'], y=df['ApplicantIncome'], ax=ax)
st.pyplot(fig)

# -------------------- Model Building --------------------
st.subheader("🤖 Loan Approval Prediction Model")

model_df = df.copy()

le = LabelEncoder()
for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']:
    model_df[col] = le.fit_transform(model_df[col])

X = model_df[['Gender','Married','Dependents','Education','Self_Employed',
              'ApplicantIncome','CoapplicantIncome','LoanAmount',
              'Loan_Amount_Term','Credit_History','Property_Area']]
y = model_df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.info(f"Model Accuracy: {accuracy:.2f}")

# -------------------- User Input --------------------
st.subheader("📝 Check Your Loan Approval Chances")

name = st.text_input("Applicant Name")
gender = st.selectbox("Gender", ['Male','Female'])
married = st.selectbox("Married", ['Yes','No'])
education = st.selectbox("Education", ['Graduate','Not Graduate'])
self_emp = st.selectbox("Self Employed", ['Yes','No'])
property_area = st.selectbox("Property Area", ['Urban','Semiurban','Rural'])
dependents = st.number_input("Dependents", 0, 5)
app_income = st.number_input("Applicant Income", 0)
co_income = st.number_input("Coapplicant Income", 0)
loan_amt = st.number_input("Loan Amount", 0)
loan_term = st.number_input("Loan Amount Term", 0)
credit = st.selectbox("Credit History", [1.0, 0.0])

if st.button("Predict Loan Status"):
    user_data = pd.DataFrame([[gender, married, dependents, education, self_emp,
                               app_income, co_income, loan_amt, loan_term, credit, property_area]],
                             columns=X.columns)

    for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
        user_data[col] = le.fit_transform(user_data[col])

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Congratulations {name}! Your loan is likely to be APPROVED")
    else:
        st.error(f"❌ Sorry {name}, your loan approval chances are LOW")

    st.write(f"Approval Probability: {probability*100:.2f}%")
