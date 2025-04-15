import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open('titanic_model.pkl', 'rb'))

st.title("🛳️ Titanic Survival Prediction App")

# User input fields
pclass = st.selectbox("Ticket Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("Number of Parents/Children Aboard", 0, 5, 0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Convert input
sex = 0 if sex == 'male' else 1
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Make prediction
X = pd.DataFrame([[pclass, sex, age, fare, sibsp, parch, embarked_q, embarked_s]],
                 columns=['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S'])
prediction = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]

if prediction == 1:
    st.success(f"🎉 This passenger is likely to **SURVIVE** (Probability: {prob:.2f})")
else:
    st.error(f"⚠️ This passenger is likely to **NOT SURVIVE** (Probability: {prob:.2f})")
