🏦 Credit Risk Prediction - German Credit Dataset

📁 crp.py

This project is a complete machine learning pipeline for predicting the creditworthiness of loan applicants using the German Credit dataset. It includes data exploration, preprocessing, model training, evaluation, and a Streamlit UI for real-time predictions.

🔍 Problem Statement

Financial institutions face challenges in assessing credit risk. This project aims to:

Classify applicants as Good or Bad credit risks

Identify key features influencing creditworthiness

Suggest improvements for the credit evaluation process

🧠 Features Used

Age

Sex

Job

Housing

Saving accounts

Checking account

Credit amount

Duration

Purpose

Engineered Feature: Debt-to-Income Ratio

📊 Project Highlights

✅ Data exploration with stats, correlations, and visualizations✅ Handling missing values & encoding categorical features✅ Feature scaling and new feature engineering✅ Model development using RandomForestClassifier + GridSearchCV✅ Performance metrics: Accuracy, Precision, Recall, F1-Score✅ Feature importance chart and confusion matrix✅ Streamlit app with custom prediction UI

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/credit-risk-prediction.git
cd credit-risk-prediction

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run crp.py

🛆 Dependencies

See requirements.txt for full list. Includes:

pandas

numpy

matplotlib

seaborn

scikit-learn

streamlit

joblib

📌 Author

Made by Vijayalakshmi for the TCS Hackathon 2025 💻

