import pandas as pd
import numpy as np

# Q1: Five functions of the pandas library with execution
print("\nQ1:")
sample_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("head():\n", sample_df.head())
print("info():"); sample_df.info()
print("describe():\n", sample_df.describe())
print("columns:\n", sample_df.columns)
print("shape:\n", sample_df.shape)

# Q2: Reindex DataFrame with index starting from 1 and incrementing by 2
print("\nQ2:")
def reindex_df(df):
    df.index = range(1, 2*len(df)+1, 2)
    return df
df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
print(reindex_df(df2))

# Q3: Sum of first 3 values in 'Values' column
print("\nQ3:")
def sum_first_three(df):
    result = df['Values'].head(3).sum()
    print("Sum of first three values:", result)
df3 = pd.DataFrame({'Values': [10, 20, 30, 40, 50]})
sum_first_three(df3)

# Q4: Word count in 'Text' column
print("\nQ4:")
def add_word_count(df):
    df['Word_Count'] = df['Text'].apply(lambda x: len(str(x).split()))
    return df
df4 = pd.DataFrame({'Text': ['Hello world', 'Pandas is great', 'Data Science']})
print(add_word_count(df4))

# Q5: Difference between .size and .shape
print("\nQ5:")
print("size(): total number of elements:", sample_df.size)
print("shape(): rows and columns:", sample_df.shape)

# Q6: Function to read Excel file
print("\nQ6:")
print("Use pd.read_excel('filename.xlsx') to read an Excel file")

# Q7: Extract usernames from emails
print("\nQ7:")
def extract_usernames(df):
    df['Username'] = df['Email'].apply(lambda x: x.split('@')[0])
    return df
df7 = pd.DataFrame({'Email': ['john.doe@example.com', 'jane.smith@gmail.com']})
print(extract_usernames(df7))

# Q8: Select rows with A > 5 and B < 10
print("\nQ8:")
def filter_rows(df):
    return df[(df['A'] > 5) & (df['B'] < 10)]
df8 = pd.DataFrame({
    'A': [3, 8, 6, 2, 9],
    'B': [5, 2, 9, 3, 1],
    'C': [1, 7, 4, 5, 2]
})
print(filter_rows(df8))

# Q9: Mean, median, and std of 'Values' column
print("\nQ9:")
def stats(df):
    print("Mean:", df['Values'].mean())
    print("Median:", df['Values'].median())
    print("Standard Deviation:", df['Values'].std())
df9 = pd.DataFrame({'Values': [10, 20, 30, 40, 50]})
stats(df9)

# Q10: 7-day moving average
print("\nQ10:")
def moving_average(df):
    df['MovingAverage'] = df['Sales'].rolling(window=7).mean()
    return df
df10 = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=10),
    'Sales': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
})
print(moving_average(df10))

# Q11: Add weekday column based on 'Date'
print("\nQ11:")
def add_weekday(df):
    df['Weekday'] = pd.to_datetime(df['Date']).dt.day_name()
    return df
df11 = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=5)})
print(add_weekday(df11))

# Q12: Select rows between 2023-01-01 and 2023-01-31
print("\nQ12:")
def select_date_range(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df[(df['Date'] >= '2023-01-01') & (df['Date'] <= '2023-01-31')]
df12 = pd.DataFrame({'Date': pd.date_range(start='2022-12-25', periods=10)})
print(select_date_range(df12))

# Q13: Necessary library to import
print("\nQ13:")
print("The first and foremost necessary library: import pandas as pd")
