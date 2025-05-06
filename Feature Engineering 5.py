# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np

# -----------------------------
# Q1: Difference between Ordinal and Label Encoding
# -----------------------------
print("Q1: Ordinal vs Label Encoding Explanation")
# Ordinal encoding: used when categories have an inherent order
ordinal_data = [['Low'], ['Medium'], ['High']]
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
ordinal_encoded = ordinal_encoder.fit_transform(ordinal_data)
print("Ordinal Encoding Example:\n", pd.DataFrame(ordinal_encoded, columns=["Priority"]))

# Label encoding: used when categories have no intrinsic order (nominal)
label_data = ['Red', 'Green', 'Blue']
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(label_data)
print("Label Encoding Example:\n", pd.DataFrame(label_encoded, columns=["Color"]))

# -----------------------------
# Q2: Target Guided Ordinal Encoding Explanation
# -----------------------------
print("\nQ2: Target Guided Ordinal Encoding Example")
# Simulate target guided encoding (e.g., Education level based on average salary)
education_levels = ['High School', 'PhD', 'Masters', 'Bachelors', 'Masters', 'PhD']
salaries = [40, 100, 80, 60, 75, 110]
df_edu = pd.DataFrame({'Education': education_levels, 'Salary': salaries})
mean_salary = df_edu.groupby('Education')['Salary'].mean().sort_values().reset_index()
mean_salary['Encoded'] = range(len(mean_salary))
edu_encoding_map = dict(zip(mean_salary['Education'], mean_salary['Encoded']))
df_edu['Encoded_Education'] = df_edu['Education'].map(edu_encoding_map)
print(df_edu)

# -----------------------------
# Q3: Covariance Explanation and Calculation
# -----------------------------
print("\nQ3: Covariance Calculation")
# Covariance quantifies the relationship between two continuous variables
data = {
    'Age': [25, 32, 47, 51, 62],
    'Income': [40000, 60000, 80000, 82000, 90000],
    'Education_Level': [1, 2, 2, 3, 3]  # assuming numeric levels for education
}
df = pd.DataFrame(data)
cov_matrix = df.cov()
print("Covariance Matrix:\n", cov_matrix)

# -----------------------------
# Q4: Label Encoding Categorical Variables
# -----------------------------
print("\nQ4: Label Encoding with scikit-learn")
df_q4 = pd.DataFrame({
    'Color': ['red', 'green', 'blue'],
    'Size': ['small', 'medium', 'large'],
    'Material': ['wood', 'metal', 'plastic']
})
le = LabelEncoder()
df_encoded_q4 = df_q4.apply(le.fit_transform)
print("Label Encoded DataFrame:\n", df_encoded_q4)

# -----------------------------
# Q5: Covariance Matrix for Given Variables
# -----------------------------
print("\nQ5: Covariance Matrix for Age, Income, and Education")
df_q5 = pd.DataFrame({
    'Age': [25, 35, 45],
    'Income': [40000, 60000, 80000],
    'Education': [1, 2, 3]
})
print("Covariance Matrix:\n", df_q5.cov())

# -----------------------------
# Q6: Encoding Method for Categorical Variables
# -----------------------------
print("\nQ6: Encoding Gender, Education Level, Employment Status")
df_q6 = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Female'],
    'Education_Level': ['High School', 'PhD', 'Bachelors'],
    'Employment_Status': ['Unemployed', 'Full-Time', 'Part-Time']
})
# Using OrdinalEncoder with assumed hierarchy
ordinal_encoder_q6 = OrdinalEncoder(categories=[
    ['Male', 'Female'],
    ['High School', 'Bachelors', 'Masters', 'PhD'],
    ['Unemployed', 'Part-Time', 'Full-Time']
])
df_q6_encoded = ordinal_encoder_q6.fit_transform(df_q6)
print("Encoded Data:\n", pd.DataFrame(df_q6_encoded, columns=df_q6.columns))

# -----------------------------
# Q7: Covariance Calculation with Mixed Variables
# -----------------------------
print("\nQ7: Covariance for Temperature, Humidity, Weather, Wind Direction")
df_q7 = pd.DataFrame({
    'Temperature': [30, 32, 35, 31, 29],
    'Humidity': [70, 65, 80, 75, 68],
    'Weather_Condition': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy'],
    'Wind_Direction': ['North', 'South', 'East', 'West', 'North']
})
# Convert categorical to numerical for covariance calc
df_q7['Weather_Condition'] = LabelEncoder().fit_transform(df_q7['Weather_Condition'])
df_q7['Wind_Direction'] = LabelEncoder().fit_transform(df_q7['Wind_Direction'])

print("Covariance Matrix:\n", df_q7.cov())
