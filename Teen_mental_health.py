import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===== 1. CẤU HÌNH GIAO DIỆN TRANG =====
st.set_page_config(
    page_title="Teen Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 2. TẢI MÔ HÌNH & BỘ CHUẨN HÓA =====
@st.cache_resource
def load_model_objects():
    # Tải mô hình Random Forest (hoặc Logistic Regression tùy kết quả train tốt nhất) và Scaler
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_objects()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_details = str(e)

# ===== 3. TIÊU ĐỀ CHÍNH =====
st.title("🧠 Teen Mental Health Prediction")
st.markdown("### Ứng dụng Học máy dự đoán sớm nguy cơ trầm cảm ở thanh thiếu niên")
st.markdown("---")

# Kiểm tra nếu chưa có file mô hình thì hiển thị hướng dẫn khắc phục nhanh
if not model_loaded:
    st.error("🚨 **KHÔNG THÌM THẤY FILE MÔ HÌNH TRONG THƯ MỤC MODELS!**")
    st.info(f"**Chi tiết lỗi:** {error_details}")
    st.markdown("""
    **Cách khắc phục:** 1. Hãy chắc chắn bạn đã tạo thư mục `models` nằm chung cấp với file script này.
    2. Chạy lại toàn bộ file Notebook để đảm bảo câu lệnh `joblib.dump()` đã xuất các file pkl vào đúng vị trí.
    """)
    st.stop()

# ===== 4. SIDEBAR - THU THẬP THÔNG TIN ĐẦU VÀO =====
st.sidebar.header("📝 Nhập Chỉ Số Kiểm Tra")

# Nhập các đặc trưng dạng số (Numeric Features)
age = st.sidebar.slider("🎂 Tuổi", min_value=13, max_value=19, value=16)
daily_social_media_hours = st.sidebar.slider("📱 Thời gian dùng MXH (giờ/ngày)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
sleep_hours = st.sidebar.slider("😴 Thời gian ngủ (giờ/ngày)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
screen_time_before_sleep = st.sidebar.slider("🌙 Thời gian xem màn hình trước ngủ (phút)", min_value=0, max_value=240, value=60, step=10)
academic_performance = st.sidebar.slider("📚 Điểm số học tập (GPA quy đổi)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
physical_activity = st.sidebar.slider("🏃 Hoạt động thể chất (giờ/tuần)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chỉ số tâm lý (Thang điểm 1 - 10)")
stress_level = st.sidebar.slider("⚡ Mức độ căng thẳng (Stress)", min_value=1, max_value=10, value=5)
anxiety_level = st.sidebar.slider("😰 Mức độ lo âu (Anxiety)", min_value=1, max_value=10, value=5)
addiction_level = st.sidebar.slider("🎮 Mức độ nghiện công nghệ", min_value=1, max_value=10, value=4)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Thông tin phân loại")
gender = st.sidebar.selectbox("👥 Giới tính", options=['female', 'male'])
platform_usage = st.sidebar.selectbox("🌐 Nền tảng MXH dùng nhiều nhất", options=['Both', 'Facebook', 'Instagram', 'None', 'TikTok', 'YouTube'])
social_interaction_level = st.sidebar.selectbox("🤝 Mức độ tương tác xã hội trực tiếp", options=['high', 'low', 'moderate'])

# ===== 5. MÀN HÌNH CHÍNH - HIỂN THỊ TỔNG QUAN =====
st.subheader("📋 Bảng thông số đối tượng hiện tại")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"• **Tuổi:** {age} | **Giới tính:** {gender}")
    st.write(f"• **Thời gian ngủ:** {sleep_hours} giờ/ngày")
with col2:
    st.write(f"• **Thời gian sử dụng MXH:** {daily_social_media_hours} giờ/ngày")
    st.write(f"• **Xem màn hình trước ngủ:** {screen_time_before_sleep} phút")
with col3:
    st.write(f"• **GPA học tập:** {academic_performance}/10")
    st.write(f"• **Tương tác xã hội trực tiếp:** {social_interaction_level.upper()}")

st.markdown("---")

# ===== 6. XỬ LÝ DỰ ĐOÁN KHI BẤM NÚT =====
if st.button("📊 BẮT ĐẦU PHÂN TÍCH NGUY CƠ", type="primary"):
    
    # Từ điển ánh xạ nhãn chữ -> số (Khớp hoàn toàn với LabelEncoder trong file huấn luyện)
    gender_map = {'female': 0, 'male': 1}
    platform_map = {'Both': 0, 'Facebook': 1, 'Instagram': 2, 'None': 3, 'TikTok': 4, 'YouTube': 5}
    social_map = {'high': 0, 'low': 1, 'moderate': 2}
    
    # Đóng gói dữ liệu đầu vào theo dạng số
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender_map[gender],
        'daily_social_media_hours': daily_social_media_hours,
        'platform_usage': platform_map[platform_usage],
        'sleep_hours': sleep_hours,
        'screen_time_before_sleep': screen_time_before_sleep,
        'academic_performance': academic_performance,
        'physical_activity': physical_activity,
        'stress_level': stress_level,
        'anxiety_level': anxiety_level,
        'addiction_level': addiction_level,
        'social_interaction_level': social_map[social_interaction_level]
    }])
    
    # Đồng bộ thứ tự chính xác của 12 cột đặc trưng gốc ban đầu
    feature_columns = [
        'age', 'gender', 'daily_social_media_hours', 'platform_usage', 
        'sleep_hours', 'screen_time_before_sleep', 'academic_performance', 
        'physical_activity', 'stress_level', 'anxiety_level', 
        'addiction_level', 'social_interaction_level'
    ]
    input_data = input_data[feature_columns]
    
    try:
        # Chuẩn hóa Z-score qua scaler
        input_scaled = scaler.transform(input_data)
        
        # Thực hiện dự đoán xác suất và phân lớp
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        
        # Hiển thị kết quả AI trực quan
        st.subheader("🔮 Kết Quả Đánh Giá Từ Hệ Thống AI")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Tỷ lệ nguy cơ trầm cảm phát hiện", value=f"{prediction_proba * 100:.1f}%")
            
        with res_col2:
            if prediction == 1 or prediction_proba >= 0.5:
                st.error("🚨 **CẢNH BÁO:** Thanh thiếu niên này thuộc nhóm **NGUY CƠ CAO** gặp phải các vấn đề về tâm lý/trầm cảm. Cần có sự quan tâm hỗ trợ từ gia đình và nhà trường.")
            else:
                st.success("🟢 **AN TOÀN:** Chỉ số đánh giá nằm trong ngưỡng **AN TOÀN / NGUY CƠ THẤP**.")
                
        # Hệ luật đưa ra khuyến nghị dựa theo hành vi người dùng nhập vào
        st.markdown("---")
        st.markdown("### 📋 Khuyến Nghị Từ Chuyên Gia Phân Tích:")
        
        recommendations = []
        if sleep_hours < 6.5:
            recommendations.append("❌ **Thời gian ngủ quá thấp (< 6.5 giờ):** Cần thiết lập lại chu kỳ sinh học, đảm bảo ngủ đủ từ 7-8 tiếng mỗi ngày để phục hồi chức năng não bộ.")
        if daily_social_media_hours > 6.0:
            recommendations.append("❌ **Thời gian sử dụng mạng xã hội quá cao (> 6 giờ/ngày):** Có nguy cơ bị quá tải thông tin tiêu cực. Nên chủ động giới hạn thời gian màn hình.")
        if screen_time_before_sleep > 60:
            recommendations.append("❌ **Sử dụng thiết bị sát giờ đi ngủ:** Ánh sáng xanh gây ức chế Melatonin làm giảm chất lượng giấc ngủ sâu. Hãy tắt thiết bị 30 phút trước khi ngủ.")
        if physical_activity < 2.0:
            recommendations.append("🏃 **Thiếu hụt hoạt động thể chất:** Nên duy trì tập thể dục hoặc vận động tối thiểu 15-30 phút mỗi ngày để tăng cường giải phóng các hormone tích cực.")
        if stress_level > 7 or anxiety_level > 7:
            recommendations.append("🩺 **Chỉ số căng thẳng/lo âu chạm ngưỡng báo động (>7/10):** Gia đình nên chủ động trò chuyện nhẹ nhàng, tránh tạo áp lực thành
