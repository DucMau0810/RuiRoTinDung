# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Tải mô hình đã lưu, scaler và label encoder
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Hàm để dự đoán
def predict_loan_status(features):
    features_scaled = scaler.transform([features])
    prediction = rf_model.predict(features_scaled)
    probability = rf_model.predict_proba(features_scaled)[:, 1]
    return prediction[0], probability[0]

# Giao diện nhập liệu cho người dùng
st.title("Dự Đoán Khả Năng Vỡ Nợ Khoản Vay")
st.write("Nhập vào các chỉ số dưới đây để dự đoán tình trạng khoản vay.")

# Nhập liệu từ người dùng
credit_policy = st.selectbox("Chính sách tín dụng (1: Approved, 0: Not Approved)", [0, 1])

# Các mục đích vay với các giá trị như trong dữ liệu
purpose = st.selectbox("Mục đích vay", [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "medical", "house", "car", "vacation", 
    "small_business", "moving", "renewable_energy"
])

int_rate = st.number_input("Lãi suất (%)", min_value=0.0, step=0.1)
installment = st.number_input("Khoản trả hàng tháng", min_value=0.0, step=50.0)
log_annual_inc = st.number_input("Log thu nhập hàng năm", min_value=0.0, step=0.1)
fico = st.number_input("Điểm FICO", min_value=0, step=1)
delinq_2yrs = st.number_input("Số lần trễ nợ trong 2 năm qua", min_value=0, step=1)
pub_rec = st.number_input("Số lần bị kiện (public record)", min_value=0, step=1)

# Mã hóa 'purpose' thành giá trị số
purpose_encoded = label_encoder.transform([purpose])[0]

# Dự đoán khi người dùng nhấn nút
if st.button("Dự Đoán"):
    # Chuẩn bị dữ liệu để dự đoán
    features = [credit_policy, purpose_encoded, int_rate, installment, log_annual_inc, fico, delinq_2yrs, pub_rec]
    prediction, probability = predict_loan_status(features)
    
    # Hiển thị kết quả
    if prediction == 1:
        st.error(f"Có nguy cơ vỡ nợ, xác suất: {probability*100:.2f}%")
    else:
        st.success(f"Không có khả năng vỡ nợ, xác suất: {probability*100:.2f}%")
