import pandas as pd

# Q1: Create a Pandas Series with given data
series1 = pd.Series([4, 8, 15, 16, 23, 42])
print("Q1: Series with given data:")
print(series1)

# Q2: Create a list with 10 elements and convert to Series
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
series2 = pd.Series(my_list)
print("\nQ2: Series from a list of 10 elements:")
print(series2)

# Q3: Create a DataFrame with given table data
data = {
    'Name': ['Alice', 'Bob', 'Claire'],
    'Age': [25, 30, 27],
    'Gender': ['Female', 'Male', 'Female']
}
df = pd.DataFrame(data)
print("\nQ3: DataFrame with Name, Age, and Gender:")
print(df)

# Q4: Difference between DataFrame and Series
print("\nQ4: DataFrame vs Series")
print("A Series is 1D labeled array, while a DataFrame is a 2D table with rows and columns.")
example_series = pd.Series([10, 20, 30])
example_df = pd.DataFrame({'Name': ['Tom', 'Jerry'], 'Age': [22, 25]})
print("Example Series:\n", example_series)
print("Example DataFrame:\n", example_df)

# Q5: Common DataFrame functions
print("\nQ5: Common DataFrame functions")
print("df.head():\n", df.head())
print("df.describe():\n", df.describe())
print("df.sort_values('Age'):\n", df.sort_values('Age'))

# Q6: Mutability of Series, DataFrame, and Panel
print("\nQ6: Mutability")
print("Series and DataFrame are mutable.")
print("Panel is deprecated in pandas and no longer recommended.")

# Q7: Create DataFrame using multiple Series
names = pd.Series(['Alice', 'Bob', 'Claire'])
ages = pd.Series([25, 30, 27])
genders = pd.Series(['Female', 'Male', 'Female'])
df_from_series = pd.DataFrame({
    'Name': names,
    'Age': ages,
    'Gender': genders
})
print("\nQ7: DataFrame created from multiple Series:")
print(df_from_series)
