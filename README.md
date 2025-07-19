# 🧠 Autism Prediction Using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Olamaryse/autism-prediction/blob/main/Autism_Prediction.ipynb)

This project uses machine learning to predict the likelihood of Autism Spectrum Disorder (ASD) based on a screening dataset. It combines data preprocessing, visualization, and classification techniques to build a robust and interpretable prediction model.

---

## 📊 Dataset Overview

The dataset contains responses to a 10-item screening questionnaire and demographic information. The target variable is `Class/ASD`, which indicates a positive or negative ASD screening result.

### 🔍 Key Features:
- `A1_Score` to `A10_Score`: Responses to AQ-10 screening questions
- `age`, `gender`, `ethnicity`, `jaundice`, `autism`: Demographic and medical history
- `result`: Aggregated score from the AQ-10
- `used_app_before`, `relation`: Screening history and test taker relationship
- `Class/ASD`: **Target** – 0 (No ASD) or 1 (ASD)

---

## 🧹 Data Preprocessing

- **Missing Value Handling**: Checked and addressed missing or null values.
- **Categorical Encoding**: Converted textual features into numerical values using label encoding.
- **Feature Selection**: Removed redundant or uninformative columns (e.g., `ID`, `age_desc`).

```python
# Sample snippet
df = df.drop(['age_desc', 'ID'], axis=1)
df['gender'] = LabelEncoder().fit_transform(df['gender'])


📈 Exploratory Data Analysis
Visualizations and correlation analysis helped identify patterns and relationships:

Distribution Plots: Examined age and AQ scores.

Correlation Matrix: Identified strongest predictors for ASD.

Class Balance: Checked for data imbalance in the target variable.

Example Insight: Patients with a family history of autism or neonatal jaundice had a higher prevalence of positive ASD screenings.

🧠 Model Building
A variety of machine learning models were tested, including:

Logistic Regression

Random Forest Classifier

Support Vector Machines

K-Nearest Neighbors

XGBoost (Extreme Gradient Boosting)

python
Copy
Edit
# Sample modeling step
model = RandomForestClassifier()
model.fit(X_train, y_train)
🧪 Evaluation Metrics
Performance was measured using:

Accuracy

Precision & Recall

F1 Score

Confusion Matrix

Key Result: The best-performing model achieved high recall, ensuring most ASD-positive cases were detected.

python
Copy
Edit
# Classification Report Example
              precision    recall  f1-score   support
           0       0.88      0.92      0.90       112
           1       0.85      0.78      0.81        61
✅ Results & Insights
XGBoost provided the best balance of performance and interpretability.

Importance plots showed that scores from AQ items were the most predictive.

Early screening combined with demographic profiling can effectively flag ASD risk.

📁 Project Structure
bash
Copy
Edit
Autism_Prediction.ipynb  # Full notebook with code, visuals, and results
README.md                # Project showcase (this file)
🚀 Future Work
Improve model performance with hyperparameter tuning.

Integrate SHAP values for model explainability.

Deploy as a web app for clinical or educational use.

💡 Takeaway
This project demonstrates how data science and machine learning can support early autism detection—an area with profound impact on public health and education.
