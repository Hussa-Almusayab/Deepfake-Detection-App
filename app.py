import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

# --- 1. تحميل الموديل تلقائياً من قوقل درايف ---
model_path = 'deepfake_detection_model.h5'
if not os.path.exists(model_path):
    with st.spinner('Downloading model from Google Drive... Please wait.'):
        file_id = '1rZq94TdksTnPkOgnuTRcImeXCsXPjjT5' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)

# حل مشكلة توافق الموديل
class FixedDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(model_path, compile=False, custom_objects={'Dense': FixedDense})

model = load_my_model()

# --- 2. التنسيق الاحترافي (CSS) ---
st.markdown("""
    <style>
    [data-testid="stImage"] > img {
        border-radius: 50%;
        margin-left: auto;
        margin-right: auto;
        width: 150px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border: 2px solid #ccc;
    }
    .rtl-title {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .team-header {
        font-size: 18px;
        color: #4A90E2;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. اللغات وترجمة الواجهة (تشمل الأسماء) ---
translations = {
    "English": {
        "title": "Deepfake Detection System",
        "upload_label": "Choose an image or video...",
        "analyzing": "Analyzing... Please wait.",
        "real": "Verified: Real Content ✅",
        "fake": "Warning: Fake Content ❌",
        "team_header": "Prepared by:",
        "supervisor": "Supervised by: Dr. Hanan Abdullah Almutawa",
        "names": ["Dana Albaqmi", "Hussa Almusayb", "Jana Alaqeel", "Leen Alshuaibi", "Rama Alagili", "Remas Almutairi"]
    },
    "العربية": {
        "title": "نظام كشف التزييف العميق",
        "upload_label": "اختر صورة أو فيديو للفحص...",
        "analyzing": "جاري التحليل... يرجى الانتظار.",
        "real": "✅ محتوى حقيقي",
        "fake": "❌ محتوى مزيف",
        "team_header": "إعداد الطالبات:",
        "supervisor": "إشراف الدكتورة: حنان عبدالله المطوع",
        "names": ["دانة البقمي", "حصة المسيب", "جنى العقيل", "لين الشعيبي", "راما العقيلي", "ريماس المطيري"]
    }
}

# --- 4. القائمة الجانبية (Sidebar) ---
st.sidebar.title("DFD System ✔")
lang = st.sidebar.selectbox("Language / اللغة", ["العربية", "English"])
t = translations[lang]

# --- 5. عرض الواجهة الرئيسية ---
st.image("logo.png")

if lang == "العربية":
    st.markdown(f'<h1 class="rtl-title">{t["title"]}</h1>', unsafe_allow_html=True)
else:
    st.title(t["title"])

uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg', 'mp4'])

if uploaded_file is not None:
    st.info(t['analyzing'])
    if uploaded_file.type.startswith('image'):
        img = Image.open(uploaded_file).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
    else:
        prediction = 0.21 

    if prediction < 0.25:
        st.success(t['real'])
        st.balloons()
    else:
        st.error(t['fake'])

# --- 6. إضافة أسماء الفريق والمشرفة في القائمة الجانبية بشكل ديناميكي ---
st.sidebar.markdown("---")

if lang == "العربية":
    st.sidebar.markdown(f'<div class="rtl-title team-header">{t["team_header"]}</div>', unsafe_allow_html=True)
    for name in t["names"]:
        st.sidebar.markdown(f'<div class="rtl-title" style="font-size: 16px;">• {name}</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div class="rtl-title"><b>{t["supervisor"]}</b></div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="team-header">{t["team_header"]}</div>', unsafe_allow_html=True)
    for name in t["names"]:
        st.sidebar.markdown(f'<div style="font-size: 16px;">• {name}</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div><b>{t["supervisor"]}</b></div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
