import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

# --- 1. تحميل الموديل تلقائياً (تأكدي من صحة الـ file_id) ---
model_path = 'deepfake_detection_model.h5'
if not os.path.exists(model_path):
    with st.spinner('Downloading model from Google Drive... Please wait.'):
        file_id = '1rZq94TdksTnPkOgnuTRcImeXCsXPjjT5' # تأكدي أن هذا الـ ID صحيح للموديل الخاص بك
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

# --- 2. التنسيق الاحترافي (CSS) - الحل السحري ---
st.markdown("""
    <style>
    /* 1. تنسيق الشعار: دائري، حجم متوسط، وظل خفيف لإزالة الزوايا البيضاء */
    [data-testid="stImage"] > img {
        border-radius: 50%; /* هذا السطر يجعل الشعار دائرياً تماماً ويقص الزوايا البيضاء */
        margin-left: auto;
        margin-right: auto;
        width: 150px !important; /* حجم احترافي للشعار */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* ظل احترافي يعطي عمقاً */
        border: 2px solid #ccc; /* إطار خفيف لضمان مظهر نظيف */
    }
    
    /* 2. تنسيق النصوص العربية لتظهر من اليمين لليسار في العنوان والواجهة */
    .rtl-title {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. اللغات وترجمة الواجهة ---
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

# --- 4. القائمة الجانبية ---
st.sidebar.title("DFD System ✔")
# اللغة الافتراضية الآن هي العربية
lang = st.sidebar.selectbox("Language / اللغة", ["العربية", "English"])
t = translations[lang]

# --- 5. عرض الواجهة الرئيسية ---

# أ. عرض الشعار (الـ CSS فوق سيتكفل بجماله)
st.image("logo.png")

# ب. عرض العنوان (مع دعم اللغة العربية)
if lang == "العربية":
    st.markdown(f'<h1 class="rtl-title">{t["title"]}</h1>', unsafe_allow_html=True)
else:
    st.title(t["title"])

# ج. صندوق رفع الملفات
uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg', 'mp4'])

if uploaded_file is not None:
    st.info(t['analyzing'])
    
    if uploaded_file.type.startswith('image'):
        # معالجة الصورة
        img = Image.open(uploaded_file).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التنبؤ
        with st.spinner('Analysing Image...'):
            prediction = model.predict(img_array)[0][0]
            
    else:
        # قيمة تجريبية للفيديو (كما طلبتم سابقاً)
        prediction = 0.21 

    # عرض النتيجة
    if prediction < 0.25:
        st.success(t['real'])
        st.balloons()
    else:
        st.error(t['fake'])

# --- إضافة أسماء الفريق والمشرفة في القائمة الجانبية ---
st.sidebar.markdown("---") # خط فاصل
st.sidebar.markdown(f'<div class="rtl-title" style="font-size: 18px; color: #4A90E2; font-weight: bold;">إعداد الطالبات:</div>', unsafe_allow_html=True)

# قائمة بأسماء الزميلات الستة (يمكنكِ تعديل الأسماء هنا)
team_members = [
    "جنى العقيل",
    "حصه المسيب",
    "دانة البقمي",
    "راما العقيلي",
    "ريماس المطيري",
    "لين الشعيبي"
]

for member in team_members:
    st.sidebar.markdown(f'<div class="rtl-title" style="font-size: 16px;">• {member}</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="rtl-title"><b>إشراف الدكتورة:</b><br>د. حنان المطوع</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
