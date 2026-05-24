import streamlit as st
import pandas as pd
import joblib

# ================= CONFIG =================
st.set_page_config(page_title="Mental Health Check", page_icon="🧠", layout="centered")

# ================= STYLE =================
st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
h1, h2, h3 {color: #38bdf8;}
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🧠 Kiểm tra sức khỏe tinh thần")
st.markdown("Ứng dụng giúp bạn **tự đánh giá nhanh trạng thái tinh thần** dựa trên thói quen sinh hoạt.")

st.warning("⚠️ Công cụ chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa.")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except:
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_model()

# ================= FORM =================
if model is None:
    st.error("❌ Chưa có model. Hãy upload file models/ lên server.")
    st.stop()

st.subheader("📋 Nhập thông tin của bạn")

with st.form("predict_form"):
    st.markdown("### 👤 Thông tin cá nhân")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Tuổi", 10, 25, 18)
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        gpa = st.slider("Điểm GPA", 0.0, 4.0, 3.0)

    with col2:
        sleep = st.slider("Ngủ (giờ/ngày)", 0.0, 12.0, 7.0)
        activity = st.slider("Vận động (giờ/tuần)", 0.0, 20.0, 3.0)

    st.markdown("### 📱 Thói quen sử dụng thiết bị")
    col3, col4 = st.columns(2)

    with col3:
        screen = st.slider("Thời gian dùng thiết bị", 0.0, 24.0, 5.0)
        social = st.slider("Thời gian mạng xã hội", 0.0, 24.0, 3.0)

    with col4:
        stress = st.slider("Mức độ stress", 0, 10, 5)
        anxiety = st.slider("Mức độ lo âu", 0, 10, 5)

    submit = st.form_submit_button("🚀 Kiểm tra ngay")

# ================= PREDICT =================
if submit:
    input_df = pd.DataFrame({
        'Age':[age], 'Gender':[gender], 'GPA':[gpa],
        'Sleep_Hours':[sleep], 'Physical_Activity':[activity],
        'Screen_Time':[screen], 'Social_Media_Hours':[social],
        'Stress_Level':[stress], 'Anxiety_Level':[anxiety]
    })

    for col, le in label_encoders.items():
        input_df[col] = le.transform([input_df[col][0]])

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Kết quả")

    st.progress(prob)

    if pred == 1:
        st.error(f"⚠️ Bạn đang có dấu hiệu cần chú ý ({prob:.1%})")
        st.markdown("👉 Hãy nghỉ ngơi nhiều hơn, giảm stress và cân nhắc nói chuyện với chuyên gia.")
    else:
        st.success(f"✅ Trạng thái ổn định ({prob:.1%})")
        st.markdown("👍 Hãy tiếp tục duy trì lối sống lành mạnh!")
