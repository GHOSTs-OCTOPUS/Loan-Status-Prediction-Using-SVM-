import streamlit as st
import numpy as np
import pandas as pd
import svm

import pickle

model = pickle.load(open('loan_status_model.pkl', 'rb'))

# Change the upper title
st.set_page_config(page_title="Loan Prediction App",page_icon="pic2.png")



def preprocess_features(features):
    # Fill missing values
    features.interpolate(method='linear', inplace=True)
    features['Gender'].fillna(features['Gender'].mode()[0], inplace=True)
    features['Married'].fillna(features['Married'].mode()[0], inplace=True)
    features['Dependents'].fillna(features['Dependents'].mode()[0], inplace=True)
    features['Self_Employed'].fillna(features['Self_Employed'].mode()[0], inplace=True)

    # Replace categorical values with numerical labels
    features.replace({'Married': {'No': 0, 'Yes': 1},
                      'Gender': {'Male': 1, 'Female': 0},
                      'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                      'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

    # Replace '3+' with 4 in the Dependents column
    features = features.replace(to_replace='3+', value=4)

    return features

def main():
    # Set a title for the web app
    # CSS styling to center-align the title
    title_style = """
        <style>
            .title {
                text-align: center;
            }
        </style>
    """

    # Display the title with centered styling
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Loan Status Prediction</h1>", unsafe_allow_html=True)
    image_path ="pic1.jpg"
    st.image(image_path,use_column_width=True)


    # Add input fields for the user to enter the feature values
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["No", "Yes"])
    dependents = st.number_input("Number of Dependents",min_value=0,max_value=4)
    education = st.selectbox("Education", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income (INR)", min_value=0, value=0)
    coapplicant_income = st.number_input("Co-applicant Income (INR)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=0)
    credit_history = st.selectbox("Credit history of individual’s repayment of their debts (0 for No history 1 for Having History) ", [0, 1])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    # Create a dictionary with the user input features
    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Preprocess the user input features
    processed_data = preprocess_features(pd.DataFrame(user_data, index=[0]))

    # Make predictions using the loaded model
    prediction = model.predict(processed_data)

    if st.button("Predict Loan Status"):
        # Make predictions using the loaded model
        prediction = model.predict(processed_data)

        # Display the prediction result
        if prediction[0] == 1:
            st.success("Congratulations! Your loan is likely to be approved.")
        else:
            st.error("Sorry, your loan is likely to be rejected.")

if __name__ == '__main__':
    main()
