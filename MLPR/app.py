import streamlit as st
import joblib
import pandas as pd

# ---------------- LOAD TRAINED FILES ----------------

model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("features.pkl")

# ---------------- LOAD RAW DATASET (FOR DROPDOWNS ONLY) ----------------

df = pd.read_csv("employees_final_dataset.csv")

# Clean NaN for dropdown safety
df['education'] = df['education'].fillna(df['education'].mode()[0])

# ---------------- TITLE ----------------

st.title("Employee Performance Prediction")
st.write("Enter Employee Details Below")

# ---------------- USER INPUT ----------------

department = st.selectbox("Department", sorted(df['department'].unique()))
region = st.selectbox("Region", sorted(df['region'].unique()))
education = st.selectbox("Education", sorted(df['education'].unique()))
gender = st.selectbox("Gender", sorted(df['gender'].unique()))
recruitment_channel = st.selectbox("Recruitment Channel", sorted(df['recruitment_channel'].unique()))

no_of_trainings = st.number_input("Number of Trainings", min_value=0)
age = st.number_input("Age", min_value=18)
previous_year_rating = st.number_input("Previous Year Rating", min_value=1, max_value=5)
length_of_service = st.number_input("Length of Service", min_value=0)

awards_won_option = st.selectbox("Awards Won", ["No", "Yes"])
awards_won = 1 if awards_won_option == "Yes" else 0

avg_training_score = st.number_input("Average Training Score", min_value=0)

# ---------------- ENCODE INPUT ----------------

department = label_encoders['department'].transform([department])[0]
region = label_encoders['region'].transform([region])[0]
education = label_encoders['education'].transform([education])[0]
gender = label_encoders['gender'].transform([gender])[0]
recruitment_channel = label_encoders['recruitment_channel'].transform([recruitment_channel])[0]

# ---------------- CREATE INPUT DATAFRAME ----------------

input_data = pd.DataFrame([[
    department,
    region,
    education,
    gender,
    recruitment_channel,
    no_of_trainings,
    age,
    previous_year_rating,
    length_of_service,
    awards_won,
    avg_training_score
]], columns=feature_names)

# ---------------- SCALE ----------------

input_data_scaled = scaler.transform(input_data)

# ---------------- PREDICT ----------------

if st.button("Predict Performance"):

    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    if prediction == 1:
        st.success("Employee is likely to Meet KPI > 80%")
    else:
        st.error("Employee is NOT likely to Meet KPI > 80%")

    st.write("Prediction Probability:", round(probability * 100, 2), "%")