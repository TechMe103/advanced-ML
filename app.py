import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained XGBoost model and scaler                                                           #cd smartPricePro1
                                                                                                  # python P1.py
with open("xgb_price_predictor.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Title
st.title("📱 SmartPhone Price Prediction with XGBoost")

# Taking manual inputs from the user
st.subheader("Enter Smartphone Features:")

model_number = st.text_input("📌 Model Number", "Enter model number")
company = st.selectbox("🏭 Company", ["Apple"])
display_size = st.number_input("📏 Display Size (inches)", min_value=4.0, max_value=7.5, step=0.1)
display_type = st.selectbox("🖥 Display Type", ["LCD", "AMOLED", "OLED"])
display_protection = st.selectbox("🛡 Display Protection", ["None", "Gorilla Glass", "Sapphire Glass"])
total_storage = st.number_input("💾 Total Storage (GB)", min_value=8, max_value=1024, step=8)
storage_type = st.selectbox("💽 Storage Type", ["HDD", "SSD", "UFS 2.1", "UFS 3.1"])
total_ram = st.number_input("🛠 Total RAM (GB)", min_value=2, max_value=32, step=1)
ram_type = st.selectbox("🔄 RAM Type", ["LPDDR4", "LPDDR5"])
processor = st.text_input("⚙️ Processor", "Enter processor name")
battery_capacity = st.number_input("🔋 Battery Capacity (mAh)", min_value=1000, max_value=7000, step=500)
front_camera = st.number_input("🤳 Front Camera MP", min_value=2, max_value=50, step=2)
front_dual_camera = st.number_input("🤳📸 Front Dual Camera MP", min_value=0, max_value=50, step=2)
back_camera = st.number_input("📷 Back Camera MP", min_value=8, max_value=200, step=8)
wide_camera = st.number_input("🔎 Wide Camera MP", min_value=0, max_value=108, step=2)
micro_camera = st.number_input("🔬 Micro Camera MP", min_value=0, max_value=20, step=2)
body_type = st.selectbox("🏗 Body Type", ["Plastic", "Metal", "Glass"])
body_back = st.selectbox("🔄 Body Back", ["Plastic", "Metal", "Glass"])
wireless_charging = st.radio("⚡ Wireless Charging", ["Yes", "No"])
nfc_support = st.radio("📡 NFC Support", ["Yes", "No"])

# Convert categorical values to numeric
wireless_charging = 1 if wireless_charging == "Yes" else 0
nfc_support = 1 if nfc_support == "Yes" else 0

# Create a DataFrame for prediction
input_data = pd.DataFrame(
    [[display_size, total_storage, total_ram, battery_capacity, front_camera, front_dual_camera,
      back_camera, wide_camera, micro_camera, wireless_charging, nfc_support]],
    columns=["Display Size", "Total Storage", "Total RAM", "Battery Capacity", 
             "Front Camera MP", "Front Dual Camera MP", "Back Camera MP",
             "Wide Camera MP", "Micro Camera MP", "Wireless Charging", "NFC Support"]
)

# Standardize input data
input_data_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)  # Ensure correct feature names
input_scaled = scaler.transform(input_data_df)


# Predict the price
if st.button("💰 Predict Price"):
    price = model.predict(input_scaled)
    st.success(f"📱 Estimated Price: **${price[0]:.2f}**")