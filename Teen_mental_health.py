import streamlit as st
import pandas as pd
import joblib
import os

# 1. Cấu hình trang
st.set_page_config(page_title="Teen Mental Health Predictor", layout="wide")
st.title("🧠 Dự đoán Nguy cơ Trầm cảm ở Thanh thiếu niên")

# 2. Tải model và scaler (Đảm bảo file nằm trong thư mục 'models')
@st.cache_resource
def load_assets():
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Lỗi tải model: {e}. Hãy kiểm tra thư mục 'models' có chứa .pkl chưa.")
    st.stop()

# 3. Sidebar nhập liệu
st.sidebar.header("📝 Thông tin thanh thiếu niên")
age = st.sidebar.number_input("Tuổi", 13, 19, 16)
gender = st.sidebar.selectbox("Giới tính", ['male', 'female'])
daily_media = st.sidebar.slider("Thời gian dùng MXH (giờ/ngày)", 0.0, 24.0, 4.0)
platform = st.sidebar.selectbox("Nền tảng phổ biến", ['Instagram', 'TikTok', 'Both', 'None', 'Facebook', 'YouTube'])
sleep = st.sidebar.slider("Thời gian ngủ (giờ/ngày)", 0.0, 12.0, 7.0)
screen_sleep = st.sidebar.slider("Màn hình trước ngủ (phút)", 0, 300, 60)
academic = st.sidebar.slider("Điểm học tập (GPA)", 0.0, 10.0, 7.0)
physical = st.sidebar.slider("Vận động (giờ/tuần)", 0.0, 20.0, 3.0)
stress = st.sidebar.slider("Căng thẳng (1-10)", 1, 10, 5)
anxiety = st.sidebar.slider("Lo âu (1-10)", 1, 10, 5)
addiction = st.sidebar.slider("Nghiện (1-10)", 1, 10, 5)
social = st.sidebar.selectbox("Tương tác xã hội", ['low', 'moderate', 'high'])

# 4. Xử lý nút bấm dự đoán
if st.button("📊 PHÂN TÍCH"):
    # Ánh xạ nhãn chữ -> số (Khớp với LabelEncoder đã train)
    gender_map = {'female': 0, 'male': 1}
    platform_map = {'Both': 0, 'Facebook': 1, 'Instagram': 2, 'None': 3, 'TikTok': 4, 'YouTube': 5}
    social_map = {'high': 0, 'low': 1, 'moderate': 2}
    
    # Tạo DataFrame đúng thứ tự 12 cột
    data = pd.DataFrame([{
        'age': age,
        'gender': gender_map[gender],
        'daily_social_media_hours': daily_media,
        'platform_usage': platform_map[platform],
        'sleep_hours': sleep,
        'screen_time_before_sleep': screen_sleep,
        'academic_performance': academic,
        'physical_activity': physical,
        'stress_level': stress,
        'anxiety_level': anxiety,
        'addiction_level': addiction,
        'social_interaction_level': social_map[social]
    }])
    
    # Chuẩn hóa và dự đoán
    try:
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]
        proba = model.predict_proba(scaled_data)[0][1]
        
        # Kết quả
        if prediction == 1:
            st.error(f"⚠️ CẢNH BÁO: Nguy cơ cao ({proba*100:.1f}%)")
        else:
            st.success(f"✅ AN TOÀN: Nguy cơ thấp ({proba*100:.1f}%)")
    except Exception as e:
        st.error(f"Lỗi dự đoán: {e}")
