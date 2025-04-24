# Credit Risk Prediction - Complete Machine Learning Pipeline
# Dataset: German Credit Data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
import joblib

# Load the dataset
data = pd.read_csv("german_credit_data.csv")

# Streamlit UI setup
st.title("Credit Risk Prediction - German Credit Data")
st.header("1. Data Exploration and Preprocessing")

# Data Exploration
st.subheader("Dataset Preview")
st.write(data.head())
st.subheader("Dataset Info")
st.text(data.info())
st.subheader("Statistical Summary")
st.write(data.describe())
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
for column in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    data[column] = le.fit_transform(data[column])

# Handle outliers (optional - log transform Credit amount if skewed)
# data['Credit amount'] = np.log1p(data['Credit amount'])

# Feature Engineering: Debt-to-Income Ratio
data['Debt_to_Income'] = data['Credit amount'] / (data['Duration'] + 1)

# Assume target is manually added for demo purposes
data['Risk'] = np.where((data['Credit amount'] < 2000) & (data['Duration'] < 24), 1, 0)

# Correlation matrix
st.subheader("Feature Correlation Matrix")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax=ax, cmap="coolwarm", annot=True)
st.pyplot(fig)

# Feature-target split
X = data[['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Debt_to_Income']]
y = data['Risk']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model Development
st.header("2. Model Training and Evaluation")
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': [100, 150], 'max_depth': [5, 10, 15]}
grid = GridSearchCV(model, params, cv=5, scoring='f1')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
st.subheader("Evaluation Metrics")
st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')
st.pyplot(fig2)

# Feature Importance
importances = best_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

st.subheader("Feature Importances")
st.bar_chart(feat_imp)

# Save the model
joblib.dump(best_model, "credit_risk_model.pkl")

# Streamlit UI for prediction
st.header("3. Try Your Own Prediction")
st.sidebar.header("Applicant Data")
def user_input_features():
    age = st.sidebar.slider('Age', 18, 75, 30)
    sex = st.sidebar.selectbox('Sex (0=Female, 1=Male)', [0, 1])
    job = st.sidebar.selectbox('Job (0-3)', [0, 1, 2, 3])
    housing = st.sidebar.selectbox('Housing (0=Own, 1=Free, 2=Rent)', [0, 1, 2])
    saving_acc = st.sidebar.selectbox('Saving Account (0-4)', [0, 1, 2, 3, 4])
    checking_acc = st.sidebar.selectbox('Checking Account (0-3)', [0, 1, 2, 3])
    credit_amount = st.sidebar.number_input('Credit Amount', value=1000)
    duration = st.sidebar.slider('Duration (in months)', 4, 72, 12)
    purpose = st.sidebar.selectbox('Purpose (0-10)', list(range(11)))

    debt_to_income = credit_amount / (duration + 1)

    data = {
        'Age': age,
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': saving_acc,
        'Checking account': checking_acc,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose,
        'Debt_to_Income': debt_to_income
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)
prediction = best_model.predict(input_scaled)

st.subheader("Prediction")
st.write("🟢 Good Credit Risk" if prediction[0] == 1 else "🔴 Bad Credit Risk")

# Optional Recommendation Section
st.header("4. Recommendations and Insights")
st.markdown("""
- **High importance features** like `Credit amount`, `Duration`, and `Debt_to_Income` should be carefully assessed.
- Automate initial screening of applicants using this model to save time and reduce defaults.
- Consider enhancing the model with additional features like employment status, credit history length, etc.
- Periodically retrain the model with new data to maintain performance.
""")
