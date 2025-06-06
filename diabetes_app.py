import streamlit as st
import numpy as np
import joblib

model = joblib.load('diabetes_xgb_model.joblib')

feature_names = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'Risk_Score'
]

st.title("Diabetes Risk Predictor (BRFSS 2015)")

st.markdown("""
Enter your health indicators below to predict your diabetes risk:
""")

user_input = {}
for feat in feature_names:
    if feat in ['BMI', 'MentHlth', 'PhysHlth']:
        user_input[feat] = st.number_input(f"{feat} (numeric)", min_value=0.0, max_value=100.0, value=25.0)
    elif feat == 'Age':
        user_input[feat] = st.number_input("Age (categorical, 1–13)", min_value=1.0, max_value=13.0, value=5.0)
    elif feat == 'GenHlth':
        user_input[feat] = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
    elif feat == 'Education':
        user_input[feat] = st.slider("Education Level (1=Never attended, 6=College grad)", 1, 6, 4)
    elif feat == 'Income':
        user_input[feat] = st.slider("Income Level (1=Lowest, 8=Highest)", 1, 8, 4)
    elif feat == 'Risk_Score':
        user_input[feat] = 0
    else:
        user_input[feat] = st.selectbox(f"{feat} (0=No, 1=Yes)", [0, 1])

risk_score_feats = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'DiffWalk']
risk_score = sum([user_input[feat] for feat in risk_score_feats])
user_input['Risk_Score'] = risk_score

input_array = np.array([user_input[feat] for feat in feature_names]).reshape(1, -1)

if st.button("Predict Diabetes Risk"):
    prediction = model.predict(input_array)[0]
    if prediction == 0:
        st.success("Prediction: No Diabetes")
    elif prediction == 1:
        st.info("Prediction: Prediabetes")
    elif prediction == 2:
        st.error("Prediction: Diabetes")
    else:
        st.warning(f"Unknown prediction: {prediction}")
