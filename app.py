
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# MUST BE FIRST
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")

# Load model, encoder, and saved columns
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('xgb_credit_model.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    columns = pickle.load(open('feature_columns.pkl', 'rb'))  # ← Uses your saved 60 columns
    return model, le, columns

model, le, feature_columns = load_artifacts()

st.title("💳 Credit Risk Assessment System")
st.markdown("**XGBoost Model • Cross-Validation Accuracy: 77.63% ± 0.31% • 60 Features**")
st.markdown("---")

page = st.sidebar.selectbox("Menu", ["Home", "Single Prediction", "Batch Prediction", "Feature Importance"])

if page == "Home":
    st.success("✅ All artifacts loaded successfully!")
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy (CV)", "77.63%")
    col2.metric("Number of Features", len(feature_columns))
    st.info("Your credit risk model is ready for real-time predictions!")

elif page == "Single Prediction":
    st.header("Predict Risk for a New Customer")

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        income = st.number_input("Monthly Income (₹)", 10000, 500000, 40000)
        total_tl = st.number_input("Total Trade Lines", 0, 50, 8)
    with col2:
        missed = st.number_input("Total Missed Payments", 0, 20, 0)
        age_oldest = st.number_input("Age of Oldest Account (months)", 0, 400, 60)
        inquiries = st.number_input("Recent Inquiries", 0, 20, 2)

    # Input data
    input_data = pd.DataFrame([{
        'Credit_Score': credit_score,
        'NETMONTHLYINCOME': income,
        'Total_TL': total_tl,
        'Tot_Missed_Pmnt': missed,
        'Age_Oldest_TL': age_oldest,
        'tot_enq': inquiries
    }])

    # CRITICAL FIX: Use saved feature_columns, NOT X
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    if st.button("Predict Risk Category", type="primary"):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        risk_label = le.inverse_transform([prediction])[0]

        st.success(f"**Predicted Risk Category: {risk_label}**")

        prob_df = pd.DataFrame({
            'Risk Level': le.classes_,
            'Probability (%)': np.round(probabilities * 100, 1)
        })

        fig = px.bar(prob_df, x='Risk Level', y='Probability (%)', color='Risk Level',
                     color_discrete_map={'P1': '#2ecc71', 'P2': '#3498db', 'P3': '#f39c12', 'P4': '#e74c3c'},
                     title="Prediction Confidence")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded = st.file_uploader("Upload customer data (CSV)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df_aligned = df.reindex(columns=feature_columns, fill_value=0)
        predictions = le.inverse_transform(model.predict(df_aligned))
        df['Predicted_Risk'] = predictions
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv, "credit_risk_predictions.csv", "text/csv")

elif page == "Feature Importance":
    st.header("Top 20 Most Important Features")
    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                 title="XGBoost Global Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Credit Risk Classification Project • December 30, 2025")