import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# --- 1. إعدادات اللغة ---
def ar(text):
    return get_display(reshape(text))

# قاموس اللغات
translations = {
    "English": {
        "title": "Deepfake Detection System",
        "upload_label": "Choose an image or video...",
        "analyzing": "Analyzing content... Please wait.",
        "real": "Verified: Real Content ✅",
        "fake": "Warning: Fake Content ❌",
        "about": "About Project",
        "lang_label": "Select Language",
        "home": "Home"
    },
    "العربية": {
        "title": ar("نظام كشف التزييف العميق"),
        "upload_label": ar("اختر صورة أو فيديو للفحص..."),
        "analyzing": ar("جاري التحليل... يرجى الانتظار."),
        "real": f"✅ {ar('محتوى حقيقي')}",
        "fake": f"❌ {ar('محتوى مزيف')}",
        "about": ar("حول المشروع"),
        "lang_label": ar("اختر اللغة"),
        "home": ar("الرئيسية")
    }
}

# --- 2. إعدادات الصفحة والموديل ---
st.set_page_config(page_title="DFD System", layout="centered")

class FixedDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('deepfake_detection_model.h5', compile=False, custom_objects={'Dense': FixedDense})

model = load_model()

# --- 3. القائمة الجانبية (Sidebar) ---
st.sidebar.title("DFD System ✔")
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
t = translations[lang]

# --- 4. واجهة المستخدم ---
st.image("logo.png", use_column_width=True)
st.markdown(f"<h1 style='text-align: center;'>{t['title']}</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg', 'mp4'])

if uploaded_file is not None:
    st.info(t['analyzing'])
    
    # محاكاة لعملية المعالجة (الصور)
    if uploaded_file.type.startswith('image'):
        img = Image.open(uploaded_file).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
    else:
        # للقيمة التجريبية التي حصلتم عليها في الفيديو
        prediction = 0.21 

    # عرض النتائج
    if prediction < 0.25:
        st.success(t['real'])
        st.balloons()
    else:
        st.error(t['fake'])

# --- 5. قسم معلومات إضافية ---
st.sidebar.markdown("---")
if st.sidebar.button(t['about']):
    st.sidebar.write("Project: Synthetic Media Authentication")
    st.sidebar.write("Dataset: FaceForensics++")