# Credit Risk Classification Project

**Multiclass Credit Risk Prediction using XGBoost**  
**Cross-Validation Accuracy: 77.63% ± 0.31%**

## Project Overview
This project automates loan approval risk classification into 4 categories:
- **P1**: Lowest Risk (Fast Approve)
- **P2**: Good Risk (Standard Approve)
- **P3**: Moderate Risk (Manual Review)
- **P4**: Highest Risk (Decline)

Built using real banking data with rigorous preprocessing, statistical feature selection, and XGBoost modeling.

## Features
- Interactive **Streamlit Dashboard** for single & batch predictions
- Feature importance visualization
- Production-ready model with saved artifacts
- 60 engineered features selected via Chi-square, ANOVA, and VIF

## How to Run the Dashboard

pip install -r requirements.txt
streamlit run app.py
Files

credit project .ipynb: Full analysis and modeling notebook
app.py: Streamlit dashboard
Model files: xgb_credit_model.pkl, label_encoder.pkl, feature_columns.pkl

Author
Trisha Paul
Live Dashboard Demo Coming Soon!