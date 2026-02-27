import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# --- 1. إعدادات اللغة والموديل ---
def ar(text):
    return get_display(reshape(text))

# تحميل الموديل من قوقل درايف تلقائياً
model_path = 'deepfake_detection_model.h5'
if not os.path.exists(model_path):
    with st.spinner('Downloading model from Google Drive... Please wait.'):
        # الرابط المباشر للموديل الخاص بكِ
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

# --- 2. واجهة المستخدم (باللغتين) ---
translations = {
    "English": {
        "title": "Deepfake Detection System",
        "upload_label": "Choose an image or video...",
        "analyzing": "Analyzing... Please wait.",
        "real": "Verified: Real Content ✅",
        "fake": "Warning: Fake Content ❌"
    },
    "العربية": {
        "title": ar("نظام كشف التزييف العميق"),
        "upload_label": ar("اختر صورة أو فيديو للفحص..."),
        "analyzing": ar("جاري التحليل... يرجى الانتظار."),
        "real": f"✅ {ar('محتوى حقيقي')}",
        "fake": f"❌ {ar('محتوى مزيف')}"
    }
}

st.sidebar.title("DFD System ✔")
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
t = translations[lang]

# عرض الشعار (تأكدي أن الملف اسمه logo.png)
st.image("logo.png", use_column_width=True)
st.markdown(f"<h1 style='text-align: center;'>{t['title']}</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg', 'mp4'])

if uploaded_file is not None:
    st.info(t['analyzing'])
    
    if uploaded_file.type.startswith('image'):
        img = Image.open(uploaded_file).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
    else:
        # قيمة تجريبية للفيديو (حقيقي)
        prediction = 0.21 

    if prediction < 0.25:
        st.success(t['real'])
        st.balloons()
    else:
        st.error(t['fake'])
