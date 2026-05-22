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
    # Tải mô hình Random Forest và Scaler từ thư mục models
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

if not model_loaded:
    st.error("🚨 **KHÔNG TÌM THẤY FILE MÔ HÌNH TRONG THƯ MỤC MODELS!**")
    st.info(f"**Chi tiết lỗi:** {error_details}")
    st.stop()

# ===== 4. SIDEBAR - THU THẬP THÔNG TIN ĐẦU VÀO =====
st.sidebar.header("📝 Nhập Chỉ Số Kiểm Tra")

# Nhập các đặc trưng dạng số (Đã điều chỉnh khớp với tập dữ liệu của bạn)
age = st.sidebar.slider("🎂 Tuổi", min_value=13, max_value=19, value=16)
daily_social_media_hours = st.sidebar.slider("📱 Thời gian dùng MXH (giờ/ngày)", min_value=1.0, max_value=8.0, value=4.5, step=0.5)
sleep_hours = st.sidebar.slider("😴 Thời gian ngủ (giờ/ngày)", min_value=4.0, max_value=9.0, value=6.5, step=0.5)
screen_time_before_sleep = st.sidebar.slider("🌙 Màn hình trước ngủ (giờ)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
academic_performance = st.sidebar.slider("📚 Điểm học tập (GPA)", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
physical_activity = st.sidebar.slider("🏃 Hoạt động thể chất (giờ/tuần)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chỉ số tâm lý (Thang điểm 1 - 10)")
stress_level = st.sidebar.slider("⚡ Căng thẳng (Stress)", min_value=1, max_value=10, value=5)
anxiety_level = st.sidebar.slider("😰 Lo âu (Anxiety)", min_value=1, max_value=10, value=6)
addiction_level = st.sidebar.slider("🎮 Nghiện công nghệ", min_value=1, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Thông tin phân loại")
# Đã đổi lại thành giao diện Tiếng Việt thân thiện
gender_ui = st.sidebar.selectbox("👥 Giới tính", options=['Nữ', 'Nam'])
platform_ui = st.sidebar.selectbox("🌐 Nền tảng MXH dùng nhiều nhất", options=['Cả hai', 'Instagram', 'TikTok'])
social_ui = st.sidebar.selectbox("🤝 Mức độ tương tác xã hội", options=['Cao', 'Thấp', 'Trung bình'])

# ===== 5. MÀN HÌNH CHÍNH - HIỂN THỊ TỔNG QUAN =====
st.subheader("📋 Bảng thông số đối tượng hiện tại")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"• **Tuổi:** {age} | **Giới tính:** {gender_ui}")
    st.write(f"• **Thời gian ngủ:** {sleep_hours} giờ/ngày")
with col2:
    st.write(f"• **Thời gian sử dụng MXH:** {daily_social_media_hours} giờ/ngày")
    st.write(f"• **Xem màn hình trước ngủ:** {screen_time_before_sleep} giờ")
with col3:
    st.write(f"• **GPA học tập:** {academic_performance}/4.0")
    st.write(f"• **Tương tác xã hội trực tiếp:** {social_ui}")

st.markdown("---")

# ===== 6. XỬ LÝ DỰ ĐOÁN KHI BẤM NÚT =====
if st.button("📊 BẮT ĐẦU PHÂN TÍCH NGUY CƠ", type="primary"):
    
    # Từ điển ánh xạ nhãn Tiếng Việt ra số (Khớp hoàn toàn với LabelEncoder trong file Notebook của bạn)
    gender_map = {'Nữ': 0, 'Nam': 1}
    platform_map = {'Cả hai': 0, 'Instagram': 1, 'TikTok': 2}
    social_map = {'Cao': 0, 'Thấp': 1, 'Trung bình': 2}
    
    # 1. Đóng gói dữ liệu đầu vào theo đúng thứ tự 12 cột lúc train
    # ['age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours', 'screen_time_before_sleep', 'academic_performance', 'physical_activity', 'social_interaction_level', 'stress_level', 'anxiety_level', 'addiction_level']
    input_df = pd.DataFrame([{
        'age': age,
        'gender': gender_map[gender_ui],
        'daily_social_media_hours': daily_social_media_hours,
        'platform_usage': platform_map[platform_ui],
        'sleep_hours': sleep_hours,
        'screen_time_before_sleep': screen_time_before_sleep,
        'academic_performance': academic_performance,
        'physical_activity': physical_activity,
        'social_interaction_level': social_map[social_ui],
        'stress_level': stress_level,
        'anxiety_level': anxiety_level,
        'addiction_level': addiction_level
    }])
    
    try:
        # 2. Chuẩn hóa Z-score qua scaler
        input_scaled = scaler.transform(input_df)
        
        # 3. Thực hiện dự đoán xác suất và phân lớp
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        
        # 4. Hiển thị kết quả AI trực quan
        st.subheader("🔮 Kết Quả Đánh Giá Từ Hệ Thống AI")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Tỷ lệ nguy cơ trầm cảm phát hiện", value=f"{prediction_proba * 100:.1f}%")
            
        with res_col2:
            # Ngưỡng cảnh báo: Cứ tỷ lệ lớn hơn 50% hoặc model nhãn 1 là báo đỏ
            if prediction == 1 or prediction_proba >= 0.5:
                st.error("🚨 **CẢNH BÁO:** Thanh thiếu niên này thuộc nhóm **NGUY CƠ CAO** gặp phải các vấn đề về tâm lý/trầm cảm. Cần có sự quan tâm hỗ trợ từ gia đình và nhà trường.")
            else:
                st.success("🟢 **AN TOÀN:** Chỉ số dự đoán cho thấy tâm lý đối tượng đang ở mức **ỔN ĐỊNH / NGUY CƠ THẤP**.")
                
        # 5. Hệ luật đưa ra khuyến nghị
        st.markdown("---")
        st.markdown("### 📋 Khuyến Nghị Từ Chuyên Gia Phân Tích:")
        
        recommendations = []
        if sleep_hours < 6.0:
            recommendations.append("❌ **Thời gian ngủ quá thấp:** Cần thiết lập lại chu kỳ sinh học, ngủ đủ từ 7-8 tiếng mỗi ngày.")
        if daily_social_media_hours > 5.0:
            recommendations.append("❌ **Dùng mạng xã hội quá nhiều:** Dễ dẫn đến quá tải thông tin tiêu cực. Nên chủ động giới hạn thời gian màn hình.")
        if screen_time_before_sleep > 2.0:
            recommendations.append("❌ **Dùng thiết bị sát giờ ngủ:** Ánh sáng xanh gây ức chế Melatonin. Hãy tắt màn hình điện tử trước khi ngủ.")
        if physical_activity < 1.0:
            recommendations.append("🏃 **Thiếu hụt hoạt động thể chất:** Nên duy trì tập thể dục tối thiểu 15-30 phút mỗi ngày để tăng cường giải phóng hormone tích cực.")
        if stress_level > 7 or anxiety_level > 7:
            recommendations.append("🩺 **Chỉ số Căng thẳng/Lo âu ở mức báo động:** Cần giảm bớt áp lực, tăng không gian chia sẻ và cân nhắc tư vấn tâm lý chuyên sâu.")
        if social_ui == 'Thấp':
            recommendations.append("🤝 **Tương tác xã hội trực tiếp thấp:** Khuyến khích tham gia sinh hoạt ngoại khóa để cải thiện kết nối cộng đồng.")

        if len(recommendations) == 0:
            st.write("✅ Mọi thói quen sinh hoạt và chỉ số đánh giá hiện tại của thanh thiếu niên đều rất lành mạnh và khoa học. Hãy tiếp tục duy trì!")
        else:
            for rec in recommendations:
                st.write(rec)
                
    except Exception as e:
        st.error(f"⚠️ Đã xảy ra lỗi hệ thống: {e}")

# ===== 7. PHẦN CHÂN TRANG =====
st.markdown("---")
st.markdown("""
<small style='color: gray;'>
⚠️ **Lưu ý độc lập:** Ứng dụng này là mô hình thực nghiệm khoa học dữ liệu phục vụ nghiên cứu tiểu luận thuật toán Học máy. Các kết quả mang tính chất xác suất tham khảo và không có giá trị thay thế cho các kết luận chẩn đoán lâm sàng từ chuyên gia y tế.
</small>
""", unsafe_utils=True)
