import pandas as pd

# Load the dataset
file_path = "ixic_data.csv"  # Replace with the path to your file
data = pd.read_csv(file_path)

# 1. Display basic information about the dataset
print("Basic Information:")
print(data.info())
print("\n")

# 2. Check for missing values
print("Missing Values Per Column:")
print(data.isnull().sum())
print("\n")

# 3. Display summary statistics for numeric columns
print("Summary Statistics:")
print(data.describe())
print("\n")

# 4. Check for duplicate rows
print("Number of Duplicate Rows:")
print(data.duplicated().sum())
print("\n")

# 5. Inspect the first and last few rows to check for anomalies
print("First Few Rows:")
print(data.head())
print("\n")
print("Last Few Rows:")
print(data.tail())
print("\n")

# 6. Check for outliers in key numeric columns using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    print(f"Outliers in column {column}:")
    outliers = detect_outliers(data, column)
    print(outliers)
    print("\n")

import matplotlib.pyplot as plt

# Plot the adjusted close price
data['Date'] = pd.to_datetime(data['Date'])
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Adj Close'], label='Adj Close')
plt.title('Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close')
plt.legend()
plt.show()

print(data.corr()['Adj Close'])


