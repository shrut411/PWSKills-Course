import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Q1: Pearson Correlation (Study Time vs Final Exam Score)
print("Q1: Pearson Correlation")
study_time = [2, 3, 4, 5, 6]
exam_score = [50, 55, 65, 70, 75]
pearson_corr, _ = pearsonr(study_time, exam_score)
print(f"Pearson Correlation: {pearson_corr:.2f}")

# -----------------------------
# Q2: Spearman Correlation (Sleep Hours vs Job Satisfaction)
print("\nQ2: Spearman Rank Correlation")
sleep_hours = [6, 7, 8, 5, 7]
job_satisfaction = [3, 4, 5, 2, 4]
spearman_corr, _ = spearmanr(sleep_hours, job_satisfaction)
print(f"Spearman Correlation: {spearman_corr:.2f}")

# -----------------------------
# Q3: Compare Pearson vs Spearman (Exercise Hours vs BMI)
print("\nQ3: Pearson and Spearman Comparison")
exercise_hours = [3, 4, 5, 6, 7]
bmi = [30, 28, 26, 25, 24]
pearson_corr_q3, _ = pearsonr(exercise_hours, bmi)
spearman_corr_q3, _ = spearmanr(exercise_hours, bmi)
print(f"Pearson: {pearson_corr_q3:.2f}, Spearman: {spearman_corr_q3:.2f}")

# -----------------------------
# Q4: Pearson (TV Time vs Physical Activity)
print("\nQ4: Pearson Correlation")
tv_hours = [2, 3, 4, 5, 6]
physical_activity = [5, 4, 3, 2, 1]
pearson_corr_q4, _ = pearsonr(tv_hours, physical_activity)
print(f"Pearson Correlation: {pearson_corr_q4:.2f}")

# -----------------------------
# Q5: Encode Categorical and Calculate Correlation (Age vs Soft Drink Preference)
print("\nQ5: Correlation Between Age and Soft Drink Preference")
data_q5 = pd.DataFrame({
    'Age': [25, 42, 37, 19, 31, 28],
    'SoftDrink': ['Coke', 'Pepsi', 'Mountain dew', 'Coke', 'Pepsi', 'Coke']
})
label_encoder = LabelEncoder()
data_q5['EncodedDrink'] = label_encoder.fit_transform(data_q5['SoftDrink'])
pearson_corr_q5, _ = pearsonr(data_q5['Age'], data_q5['EncodedDrink'])
print(data_q5)
print(f"Pearson Correlation: {pearson_corr_q5:.2f}")

# -----------------------------
# Q6: Sales Calls vs Sales Made
print("\nQ6: Pearson Correlation (Sales Calls vs Sales Made)")
sales_calls = [20, 25, 30, 35, 40]
sales_made = [4, 6, 7, 10, 12]
pearson_corr_q6, _ = pearsonr(sales_calls, sales_made)
print(f"Pearson Correlation: {pearson_corr_q6:.2f}")
