import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import stats

# Initial DataFrame
course_name = ['Data Science', 'Machine Learning', 'Big Data', 'Data Engineer']
duration = [2, 3, 6, 4]
df = pd.DataFrame(data={'course_name': course_name, 'duration': duration})

# Q1: Print second row of df
print("Q1: Second row of df")
print(df.iloc[1])

# Q2: Difference between loc and iloc
print("\nQ2: loc vs iloc")
print("loc uses labels (e.g., df.loc[1]), iloc uses integer indices (e.g., df.iloc[1])")

# Q3: Reindexing and accessing elements
reindex = [3, 0, 1, 2]
new_df = df.reindex(reindex)
print("\nQ3: new_df.loc[2] and new_df.iloc[2]")
print("new_df.loc[2]:\n", new_df.loc[2])
print("new_df.iloc[2]:\n", new_df.iloc[2])

# Q4: Statistical measures on df1
columns = ['column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6']
indices = [1, 2, 3, 4, 5, 6]
df1 = pd.DataFrame(np.random.rand(6, 6), columns=columns, index=indices)

print("\nQ4:")
print("Mean of each column:\n", df1.mean())
print("Std Dev of 'column_2':", df1['column_2'].std())

# Q5: Replacing value with string and checking mean
print("\nQ5:")
df1.loc[2, 'column_2'] = 'hello'
try:
    print("Mean of column_2:", df1['column_2'].mean())
except Exception as e:
    print("Error:", e)
    print("Explanation: Cannot calculate mean with non-numeric data")

# Q6: Window functions in Pandas
print("\nQ6:")
print("Window functions perform calculations across a set of rows.\nTypes: rolling(), expanding(), ewm()")

# Q7: Print current month and year
print("\nQ7: Current month and year")
today = pd.Timestamp.today()
print(today.strftime("%B %Y"))

# Q8: Difference between two dates
print("\nQ8:")
date1 = pd.to_datetime(input("Enter the first date (YYYY-MM-DD): "))
date2 = pd.to_datetime(input("Enter the second date (YYYY-MM-DD): "))
delta = abs(date2 - date1)
print(f"Days: {delta.days}, Hours: {delta.total_seconds() // 3600}, Minutes: {delta.total_seconds() // 60}")

# Q9: Convert column to categorical type
print("\nQ9:")
file_path = input("Enter CSV file path with categorical data: ")
col_name = input("Enter the column name to convert to category: ")
category_order = input("Enter category order (comma-separated): ").split(',')

df_cat = pd.read_csv(file_path)
df_cat[col_name] = pd.Categorical(df_cat[col_name], categories=category_order, ordered=True)
print("Sorted Data:\n", df_cat.sort_values(by=col_name))

# Q10: Stacked bar chart for product sales
print("\nQ10:")
sales_file = input("Enter CSV file path for sales data: ")
sales_df = pd.read_csv(sales_file)
pivot_df = sales_df.pivot(index='Date', columns='Product', values='Sales')
pivot_df.plot(kind='bar', stacked=True)
plt.title('Sales by Product Category Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(title='Product')
plt.tight_layout()
plt.show()

# Q11: Student stats table
print("\nQ11:")
student_file = input("Enter student data CSV path: ")
student_df = pd.read_csv(student_file)
mean_score = student_df['Test Score'].mean()
median_score = student_df['Test Score'].median()
mode_score = student_df['Test Score'].mode().tolist()

print("\n+-----------+--------+")
print("| Statistic | Value  |")
print("+-----------+--------+")
print(f"| Mean      | {mean_score:.2f} |")
print(f"| Median    | {median_score}    |")
print(f"| Mode      | {', '.join(map(str, mode_score))} |")
print("+-----------+--------+")
