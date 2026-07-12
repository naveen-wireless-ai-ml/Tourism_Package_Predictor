import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="naveenaggarwal1989/tourism-package-model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase a tourism package
based on their demographic and interaction data. Please provide the details below.
""")

# User input widgets for each feature
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=90, value=30, step=1)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier (1=Highest, 3=Lowest)", [1, 2, 3])
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=10, step=1)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=20, value=2, step=1)
    numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferredpropertystar = st.selectbox("Preferred Property Star Rating", [1.0, 2.0, 3.0, 4.0, 5.0])
    maritalstatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    numberoftrips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=1, step=1)
    passport_input = st.selectbox("Passport Holder", ["Yes", "No"])
    pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score (1=Low, 5=High)", [1, 2, 3, 4, 5])
    owncar_input = st.selectbox("Owns a Car", ["Yes", "No"])
    numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=10, value=0, step=1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Others"])
    monthlyincome = st.number_input("Monthly Income", min_value=0, max_value=200000, value=25000, step=1000)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Map 'Yes'/'No' to 1/0 for Passport and OwnCar
        passport = 1 if passport_input == "Yes" else 0
        owncar = 1 if owncar_input == "Yes" else 0

        # Assemble input into DataFrame, ensuring correct column order and types for the model
        input_data = pd.DataFrame([{
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'NumberOfFollowups': numberoffollowups,
            'ProductPitched': productpitched,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'Designation': designation,
            'MonthlyIncome': monthlyincome
        }])

        # Ensure the column order matches the training data features (Xtrain)
        # This assumes Xtrain has the same columns and order as the model expects.
        # If Xtrain.columns is not directly accessible, you might need to infer the order
        # from the model's preprocessing step or a sample Xtrain.
        # For simplicity, assuming the order from input_data creation is correct
        # as long as all features are included and named correctly.

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[:, 1][0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"The model predicts: **Customer WILL purchase the package** (Probability: {prediction_proba:.2f})")
        else:
            st.info(f"The model predicts: **Customer WILL NOT purchase the package** (Probability: {prediction_proba:.2f})")
