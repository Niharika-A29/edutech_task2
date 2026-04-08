import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = sns.load_dataset('titanic')
print("Dataset loaded. Shape:", df.shape) 
print("\nMissing values before cleaning:")
print(df.isnull().sum()) 

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Values Heatmap - yellow = Missing")
plt.tight_layout()
plt.savefig("missing_values_heatmap.png")
plt.show()
print("Heatmap saved!")

print("\nDropping 'deck' column - 77% missing")
df = df.drop(columns=['deck'])
print("'deck' column removed.")

median_age = df['age'].median()
print(f"\nMedian age : {median_age}")
df["age"] = df["age"].fillna(median_age)
print(f"Missing age values after fix: {df['age'].isnull().sum()}")

embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
print(f"\nFilled 'embarked' with mode: {embarked_mode}")
print(f"Missing embarked values after fix: {df['embarked'].isnull().sum()}")

imputer = SimpleImputer(strategy='median')
df[['age', 'fare']] = imputer.fit_transform(df[['age', 'fare']])
print("\nSimpleImputer applied to age and fare.")
print(f"Missing age: {df['age'].isnull().sum()}")
print(f"Missing fare: {df['fare'].isnull().sum()}")

print("\nFare statistics BEFORE outlier treatment:")
print(df['fare'].describe())

Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

print(f"\nQ1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower limit: {lower_limit}, Upper limit: {upper_limit}")

df['fare'] = df['fare'].clip(lower=lower_limit, upper=upper_limit)

print("\nFare statistics AFTER outlier treatment:")
print(df['fare'].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=df['fare'], ax=axes[0])
axes[0].set_title("Fare After Outlier Treatment")

sns.boxplot(y=df['age'], ax=axes[1])
axes[1].set_title("Age Distribution")

plt.tight_layout()
plt.savefig("boxplots.png")
plt.show()
print("Boxplots saved!")

print("\n--- VERIFICATION ---")
print("Missing values AFTER cleaning:")
print(df.isnull().sum())
print(f"\nFinal shape: {df.shape}")

df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")