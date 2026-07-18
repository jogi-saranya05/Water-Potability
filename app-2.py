import streamlit as st
import pandas as pd
import shap
import google.generativeai as genai
from sklearn.ensemble import GradientBoostingClassifier
import pickle

st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Water Potability Predictor")

st.markdown("""
### AI-Powered Water Quality Assessment

Predict whether water is safe for drinking using Machine Learning.
""")

st.divider()

st.sidebar.header("ℹ️ About")

st.sidebar.info("""
This app predicts whether water is safe to drink.

Model:
- Gradient Boosting

Features:
- ML Prediction
- SHAP Explainability
- Gemini AI Report
""")
# Load saved model, scaler, and explainer
with open('final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_genai = genai.GenerativeModel('gemini-flash-latest')

# Input fields
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness", 0.0, 500.0, 180.0)
    solids = st.number_input("Solids", 0.0, 60000.0, 20000.0)
    chloramines = st.number_input("Chloramines", 0.0, 15.0, 7.0)
    sulfate = st.number_input("Sulfate", 0.0, 500.0, 330.0)

with col2:
    conductivity = st.number_input("Conductivity", 0.0, 800.0, 420.0)
    organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 14.0)
    trihalomethanes = st.number_input("Trihalomethanes", 0.0, 120.0, 66.0)
    turbidity = st.number_input("Turbidity", 0.0, 10.0, 4.0)

if st.button("Predict Potability"):

    with st.spinner("Analyzing Water Sample..."):

        input_data = pd.DataFrame(
            [[ph, hardness, solids, chloramines, sulfate,
              conductivity, organic_carbon, trihalomethanes, turbidity]],
            columns=[
                'ph','Hardness','Solids','Chloramines','Sulfate',
                'Conductivity','Organic_carbon','Trihalomethanes','Turbidity'
            ]
        )

        input_scaled = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

        prediction = final_model.predict(input_scaled_df)[0]
        prob = final_model.predict_proba(input_scaled_df)[0]
        confidence = max(prob) * 100

        verdict = "✅ Potable (Safe)" if prediction == 1 else "❌ Not Potable (Unsafe)"

        # SHAP
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(input_scaled_df)

        feature_impacts = list(zip(
            input_data.columns,
            input_data.iloc[0].values,
            shap_values[0]
        ))

        feature_impacts.sort(key=lambda x: abs(x[2]), reverse=True)
        top_factors = feature_impacts[:4]

        factors_text = "\n".join([
            f"- {name}: value={value:.2f}, impact={'increases' if impact>0 else 'decreases'} potability"
            for name, value, impact in top_factors
        ])

        prompt = f"""
Prediction: {verdict}

Top contributing factors:
{factors_text}

Write a simple 3-4 sentence explanation for a normal user.
"""

        response = model_genai.generate_content(prompt)
        report = response.text

    tab1, tab2, tab3 = st.tabs(
        ["Prediction", "AI Report", "Explainability"]
    )

    with tab1:

        if prediction == 1:
            st.success(verdict)
        else:
            st.error(verdict)

        st.metric("Confidence", f"{confidence:.2f}%")

    with tab2:

        st.write(report)

        st.download_button(
            "📄 Download Report",
            report,
            file_name="water_report.txt"
        )

    with tab3:

        st.write("### Top Contributing Factors")

        for name, value, impact in top_factors:

            direction = "⬆️ increases" if impact > 0 else "⬇️ decreases"

            st.write(
                f"**{name}** : {value:.2f} ({direction} potability)"
            )
