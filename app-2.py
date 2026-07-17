import streamlit as st
import pandas as pd
import shap
import google.generativeai as genai
from sklearn.ensemble import GradientBoostingClassifier
import pickle

st.title("💧 Water Potability Predictor")
st.write("Enter water test values below to check potability, see why, and get a plain-English report.")

# Load saved model, scaler, and explainer
with open('final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_genai = genai.GenerativeModel('gemini-flash-latest')

# Input fields
ph = st.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness", 0.0, 500.0, 180.0)
solids = st.number_input("Solids", 0.0, 60000.0, 20000.0)
chloramines = st.number_input("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.number_input("Sulfate", 0.0, 500.0, 330.0)
conductivity = st.number_input("Conductivity", 0.0, 800.0, 420.0)
organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 14.0)
trihalomethanes = st.number_input("Trihalomethanes", 0.0, 120.0, 66.0)
turbidity = st.number_input("Turbidity", 0.0, 10.0, 4.0)

if st.button("Predict Potability"):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                                 conductivity, organic_carbon, trihalomethanes, turbidity]],
                               columns=['ph','Hardness','Solids','Chloramines','Sulfate',
                                        'Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])

    input_scaled = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

    prediction = final_model.predict(input_scaled_df)[0]
    verdict = "✅ Potable (Safe)" if prediction == 1 else "❌ Not Potable (Unsafe)"
    st.subheader(verdict)

    # SHAP explanation
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(input_scaled_df)

    feature_impacts = list(zip(input_data.columns, input_data.iloc[0].values, shap_values[0]))
    feature_impacts.sort(key=lambda x: abs(x[2]), reverse=True)
    top_factors = feature_impacts[:4]

    factors_text = "\n".join([
        f"- {name}: value = {value:.2f}, impact = {'increases' if impact > 0 else 'decreases'} potability likelihood"
        for name, value, impact in top_factors
    ])

    st.write("**Top contributing factors:**")
    for name, value, impact in top_factors:
        direction = "⬆️ increases" if impact > 0 else "⬇️ decreases"
        st.write(f"- {name}: {value:.2f} ({direction} potability likelihood)")

    # GenAI report
    prompt = f"""
You are a water quality assistant. Based on the following machine learning model output, write a short, plain-English report (3-4 sentences) for a non-technical reader, such as a water treatment operator.

Prediction: {verdict}

Top contributing factors:
{factors_text}

Explain in simple terms why the water was classified this way, and give one practical recommendation if it was classified as unsafe. Keep it concise and avoid technical jargon.
"""
    response = model_genai.generate_content(prompt)
    st.write("**Plain-English Report:**")
    st.write(response.text)
