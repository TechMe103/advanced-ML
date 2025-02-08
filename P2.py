import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb 
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("apple_only_devices_dataset.csv")

# Check for missing values and drop them
data.dropna(inplace=True)

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Apply Label Encoding for categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    if data[col].nunique() <= 10:  # Label Encoding for fewer unique values
        data[col] = label_encoder.fit_transform(data[col])
    else:  # One-Hot Encoding for larger unique value categories
        data = pd.get_dummies(data, columns=[col], drop_first=True)

# Creating price categories (for analysis, not model training)
bins = [0, 300, 600, 900, 1200, np.inf]
labels = ['Budget', 'Mid-Range', 'Upper Mid-Range', 'Premium', 'Ultra-Premium']
data['price_range'] = pd.cut(data['Price (USD)'], bins=bins, labels=labels)

# Select features and target variable
X = data.drop(columns=["Price (USD)", "price_range"])  # Remove price column
y = data["Price (USD)"]  # Target variable

# Split into train-test set (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print Evaluation Metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared Score (R²): {r2}")

# Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.title("Actual vs Predicted Price (XGBoost)")
plt.show()

import pickle

# Save the trained XGBoost model
with open("xgb_price_predictor.pkl", "wb") as file:
    pickle.dump(xgb_model, file)

# Save the scaler as well (to use during prediction)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model and Scaler saved successfully as pickle files!")


