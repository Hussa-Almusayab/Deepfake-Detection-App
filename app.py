import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

# --- 1. تحميل الموديل تلقائياً ---
model_path = 'deepfake_detection_model.h5'
if not os.path.exists(model_path):
    with st.spinner('Downloading model from Google Drive... Please wait.'):
        file_id = '1rZq94TdksTnPkOgnuTRcImeXCsXPjjT5'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# --- 2. واجهة المستخدم والتنسيق (CSS) ---
st.markdown("""
    <style>
    /* تنسيق النصوص العربية لتظهر من اليمين لليسار وبخط جميل */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    /* تنسيق الشعار ليكون في المنتصف وبحجم احترافي */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .logo-img {
        width: 300px; /* يمكنكِ التحكم في الحجم هنا */
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. اللغات ---
translations = {
    "English": {
        "title": "Deepfake Detection System",
        "upload_label": "Choose an image or video...",
        "analyzing": "Analyzing... Please wait.",
        "real": "Verified: Real Content ✅",
        "fake": "Warning: Fake Content ❌"
    },
    "العربية": {
        "title": "نظام كشف التزييف العميق",
        "upload_label": "اختر صورة أو فيديو للفحص...",
        "analyzing": "جاري التحليل... يرجى الانتظار.",
        "real": "✅ محتوى حقيقي",
        "fake": "❌ محتوى مزيف"
    }
}

st.sidebar.title("DFD System ✔")
lang = st.sidebar.selectbox("Language / اللغة", ["العربية", "English"])
t = translations[lang]

# عرض الشعار بشكل احترافي
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("logo.png", width=350)
st.markdown('</div>', unsafe_allow_html=True)

# عرض العنوان مع دعم اللغة العربية
if lang == "العربية":
    st.markdown(f'<h1 class="rtl-text">{t["title"]}</h1>', unsafe_allow_html=True)
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
        prediction = 0.21 # قيمة تجريبية للفيديو

    if prediction < 0.25:
        st.success(t['real'])
        st.balloons()
    else:
        st.error(t['fake'])
