import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
data = pd.read_csv("apple_only_devices_dataset.csv")

# Display first few rows
print(data.head())

# Check dataset information
print(data.info())

# Check for missing values
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.isnull().sum())

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols)

# Apply Label Encoding or One-Hot Encoding based on the number of unique values
label_encoder = LabelEncoder()
for col in categorical_cols:
    if data[col].nunique() <= 10:  # Apply Label Encoding for fewer unique values
        data[col] = label_encoder.fit_transform(data[col])
    else:  # Apply One-Hot Encoding for larger unique value categories
        data = pd.get_dummies(data, columns=[col], drop_first=True)

print("Updated Data after Encoding:")
print(data.head())


sns.histplot(data['Price (USD)'], kde=True, bins=20, color="blue")
plt.title("Distribution of Phone Prices (USD)")
plt.xlabel("Price (USD)")
plt.ylabel("Count")
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Creating price categories
bins = [0, 300, 600, 900, 1200, np.inf]  # Define price ranges
labels = ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Premium', 'Ultra-Premium']
data['price_range'] = pd.cut(data['Price (USD)'], bins=bins, labels=labels)

# RAM vs Price Range
plt.figure(figsize=(8,5))
sns.boxplot(x=data["price_range"], y=data["Total RAM"], palette="coolwarm")
plt.title("RAM vs Price Range")
plt.xlabel("Price Range")
plt.ylabel("Total Ram")
plt.show()

# Battery Power vs Price Range
plt.figure(figsize=(8,5))
sns.boxplot(x=data["price_range"], y=data["Battery Capacity"], palette="coolwarm")
plt.title("Battery Power vs Price Range")
plt.xlabel("Price Range")
plt.ylabel("Battery Power")
plt.show()

# Screen Size vs Price Range
plt.figure(figsize=(8,5))
sns.boxplot(x=data["price_range"], y=data["Display Size"], palette="coolwarm")
plt.title("Screen Size vs Price Range")
plt.xlabel("Price Range")
plt.ylabel("Display Size")
plt.show()

