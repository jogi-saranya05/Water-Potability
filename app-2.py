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

# ---------------------------------------------------------
# Cache model, scaler, and SHAP explainer so they load once
# per session instead of on every button click / rerun
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

final_model, scaler = load_model_and_scaler()
explainer = load_explainer(final_model)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_genai = genai.GenerativeModel('gemini-flash-latest')

# ---------------------------------------------------------
# Preset sample buttons
# ---------------------------------------------------------
st.write("**Quick test samples:**")
preset_col1, preset_col2, preset_col3 = st.columns(3)

if "preset" not in st.session_state:
    st.session_state.preset = None

with preset_col1:
    if st.button("💧 Load Safe Sample"):
        st.session_state.preset = "safe"
with preset_col2:
    if st.button("⚠️ Load Unsafe Sample"):
        st.session_state.preset = "unsafe"
with preset_col3:
    if st.button("🔄 Reset"):
        st.session_state.preset = None

presets = {
    "safe":   {"ph": 7.2, "hardness": 150, "solids": 15000, "chloramines": 5.0,
               "sulfate": 250, "conductivity": 400, "organic_carbon": 10,
               "trihalomethanes": 50, "turbidity": 3.0},
    "unsafe": {"ph": 3.5, "hardness": 350, "solids": 45000, "chloramines": 12,
               "sulfate": 450, "conductivity": 700, "organic_carbon": 25,
               "trihalomethanes": 100, "turbidity": 8.5}
}

default = presets.get(st.session_state.preset, {})

# ---------------------------------------------------------
# Input fields — now actually wired to presets via `default`
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0,
                          value=default.get("ph", 7.0), step=0.1)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0,
                                value=default.get("hardness", 180.0), step=1.0)
    solids = st.number_input("Solids (ppm)", min_value=0.0, max_value=60000.0,
                              value=default.get("solids", 20000.0), step=100.0)
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0,
                                   value=default.get("chloramines", 7.0), step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0,
                               value=default.get("sulfate", 330.0), step=1.0)

with col2:
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=800.0,
                                    value=default.get("conductivity", 420.0), step=1.0)
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0,
                                      value=default.get("organic_carbon", 14.0), step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=120.0,
                                       value=default.get("trihalomethanes", 66.0), step=1.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0,
                                 value=default.get("turbidity", 4.0), step=0.1)

if st.button("Predict Potability"):

    with st.spinner("Analyzing Water Sample..."):

        input_data = pd.DataFrame(
            [[ph, hardness, solids, chloramines, sulfate,
              conductivity, organic_carbon, trihalomethanes, turbidity]],
            columns=[
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]
        )

        input_scaled = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

        prediction = final_model.predict(input_scaled_df)[0]
        prob = final_model.predict_proba(input_scaled_df)[0]
        confidence = max(prob) * 100

        verdict = "✅ Potable (Safe)" if prediction == 1 else "❌ Not Potable (Unsafe)"

        # SHAP (explainer now loaded once via cache, not rebuilt here)
        shap_values = explainer.shap_values(input_scaled_df)

        feature_impacts = list(zip(
            input_data.columns,
            input_data.iloc[0].values,
            shap_values[0]
        ))

        feature_impacts.sort(key=lambda x: abs(x[2]), reverse=True)
        top_factors = feature_impacts[:4]

        factors_text = "\n".join([
            f"- {name}: value={value:.2f}, impact={'increases' if impact > 0 else 'decreases'} potability"
            for name, value, impact in top_factors
        ])

        prompt = f"""
Prediction: {verdict}

Top contributing factors:
{factors_text}

Write a simple 3-4 sentence explanation for a normal user.
"""

        # Gemini call wrapped so a rate-limit / API error doesn't crash the app
        try:
            response = model_genai.generate_content(prompt)
            report = response.text
        except Exception as e:
            st.warning("⚠️ AI report generation is unavailable right now (API limit or connection issue). Showing key factors instead.")
            report = "Key factors affecting this prediction:\n\n" + factors_text

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
