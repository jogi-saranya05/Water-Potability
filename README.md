<img width="1440" height="804" alt="image" src="https://github.com/user-attachments/assets/9d7eabfd-771a-4982-9415-5869aa932277" /> # 💧 Water Potability Predictor

An AI-powered web application that predicts whether water is safe for drinking based on key water quality parameters. The application combines a Machine Learning model with Explainable AI (SHAP) and Google's Gemini AI to provide both predictions and easy-to-understand explanations.

## 🚀 Live Demo

🔗 **Streamlit App:** https://water-potability-aipredictor.streamlit.app/

---

## 📌 Features

- 💧 Predicts whether water is **Potable** or **Not Potable**
- 🤖 AI-generated explanation using **Google Gemini**
- 📊 Prediction confidence score
- 📈 SHAP-based feature importance for model explainability
- 📄 Downloadable water quality report
- 🎨 Responsive and user-friendly Streamlit interface

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **Model:** Gradient Boosting Classifier
- **Explainable AI:** SHAP
- **Generative AI:** Google Gemini API
- **Data Processing:** Pandas, NumPy
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
Water-Potability-Predictor/
│
├── app.py
├── final_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

---

## 📊 Input Parameters

The model predicts water potability using the following water quality parameters:

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

---

## ⚙️ How It Works

1. Enter the water quality parameters.
2. The data is scaled using the trained StandardScaler.
3. The Gradient Boosting model predicts whether the water is potable.
4. SHAP identifies the most influential features behind the prediction.
5. Gemini AI generates a plain-English explanation of the result.
6. Users can download the generated report.

---

## 📈 Machine Learning Model

- **Algorithm:** Gradient Boosting Classifier
- **Preprocessing:** StandardScaler
- **Explainability:** SHAP TreeExplainer

The model was trained on the **Water Potability Dataset**, which contains water quality measurements and corresponding potability labels.

---



## 📷 Screenshots

### Home Page

 <img width="1440" height="811" alt="image" src="https://github.com/user-attachments/assets/2e1996f8-954b-4fdf-969d-9cc1f04ea1cc" />


### Prediction Result

<img width="1437" height="810" alt="image" src="https://github.com/user-attachments/assets/96ce83db-b9c0-41f9-9539-f2cc2e3cf6b1" />


### AI Report

 <img width="1440" height="804" alt="image" src="https://github.com/user-attachments/assets/2e9f37b4-ac78-4fa4-b57d-fccd746f45ae" />


---

## 📌 Future Improvements

- Interactive visualizations and comparison charts
- Additional machine learning models for comparison
- PDF report generation
- Historical prediction tracking
- Water quality recommendations based on WHO standards

---

## 👩‍💻 Author

**Saranya Jogi, Risha Rastogi and Yosha Singh**

GitHub: https://github.com/jogi-saranya05

LinkedIn: www.linkedin.com/in/saranya-jogi-140531340

---

## 📄 License

This project is intended for educational and demonstration purposes.
