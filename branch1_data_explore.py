# branch: data-loading-exploration

import pandas as pd

# 1. Load dataset
df = pd.read_csv("insurance_data_linear.csv")

# 2. Basic info
print(df.head())
print(df.info())
print(df.describe())

# 3. Check missing values
print(df.isnull().sum())

# 4. Simple EDA
print(df["sex"].value_counts())
print(df["smoker"].value_counts())
print(df["region"].value_counts())

import matplotlib.pyplot as plt

# Calculate mean and median
mean_val = df["charges"].mean()
median_val = df["charges"].median()

plt.figure(figsize=(12, 7))
plt.hist(df["charges"], bins=40, color='green')
plt.axvline(median_val, color='orange', linestyle='solid', linewidth=2, label=f'Median: {median_val:.2f}')
plt.axvline(mean_val, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')

plt.xlabel("Insurance Charges")
plt.ylabel("Frequency")
plt.title("Spread of Medical Insurance Costs")
plt.savefig("Graph.png")
