import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Aseel AI - Disease Diagnostic", page_icon="🐓", layout="centered")

# Custom Professional Styling (CSS)
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .header-box { text-align: center; padding: 15px 0; }
    .alert-card {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 14px;
        border-radius: 8px;
        margin: 12px 0;
        color: #881337;
        font-weight: 500;
    }
    .success-card {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 14px;
        border-radius: 8px;
        margin: 12px 0;
        color: #14532d;
        font-weight: 500;
    }
    .med-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Header with Logo (ASEEL.png)
col_logo, col_title = st.columns([1, 4])
with col_logo:
    logo_path = os.path.join(os.path.dirname(__file__), "ASEEL.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.write("🐓")

with col_title:
    st.markdown("<h2 style='margin-bottom:0;'>Aseel Disease Diagnostic AI</h2>", unsafe_allow_html=True)
    st.caption("AI-Powered Poultry Health Monitor | Live Camera & Gallery Scanner")

st.divider()

# 2. Model & Haar Cascade Load
@st.cache_resource
def load_resources():
    model_path = os.path.join(os.path.dirname(__file__), "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    return interpreter, face_cascade

interpreter, face_cascade = load_resources()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ['Coccidiosis', 'Healthy', 'New Castle Disease (Rani Khet)', 'Salmonella']

TREATMENTS = {
    'Coccidiosis': {
        'title': 'Khuni / Surkh Beeth (Coccidiosis)',
        'alert': 'Murghay ko fauri alag darbay mein band karein aur zameen khushk rakhein.',
        'symptoms': 'Surkh ya khoon aalood beeth, chehray ka peela parna, par latkana.',
        'care': 'Geeli mitti ya gandi jagah se door rakhein, thanda paani mat dein.',
        'desi': 'Saib ka sirka (5ml fi liter) aur lehsan ka paani dein.',
        'medical': 'Vet Amprolium powder ya Baycox (1ml fi liter paani mein 3-5 din).'
    },
    'New Castle Disease (Rani Khet)': {
        'title': 'Sabz Beeth / Lakwa Jhatka (Rani Khet)',
        'alert': 'Shadeed khatarnak virus! Doosre parindon se fauri door karein.',
        'symptoms': 'Sabz patli beeth, gardan ka ghumna, jhatkay lagna.',
        'care': 'Darbay ko garam rakhein aur jaraseem-kash spray karein.',
        'desi': 'Gur, sonf aur ajwain ka garam qehwa bana kar dein.',
        'medical': 'Neurobion drops taqat ke liye aur Enrofloxacin antibiotic dein.'
    },
    'Salmonella': {
        'title': 'Safaid / Chuna Beeth (Salmonella)',
        'alert': 'Bartan saaf karein aur taza ubla hua paani dein.',
        'symptoms': 'Peechay safaid chuna jaisi beeth ka lagna, bhook na lagna.',
        'care': 'Purana daana ya alooda paani hargiz na dein.',
        'desi': 'Ublay anday ki zardi mein haldi mila kar dein.',
        'medical': 'Gentamicin drops ya Oxytetracycline powder 4-5 din dein.'
    },
    'Healthy': {
        'title': 'Sehatmand Parinda (Healthy)',
        'alert': 'MashaAllah parinda bilkul theek hai.',
        'symptoms': 'Bandhi hui theek beeth, koi alamat nahi.',
        'care': 'Rozana taza paani aur saaf daana jari rakhein.',
        'desi': 'Koe zaroorat nahi.',
        'medical': 'Multivitamins routine mein dein.'
    }
}

# Image Validation (Human Face & Dropping Check)
def validate_dropping_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) > 0:
        return False, "human_face"
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    brightness_variance = np.var(v_channel)
    h_channel = hsv[:, :, 0]
    unique_colors = len(np.unique(h_channel))
    
    if brightness_variance < 25 or unique_colors < 12:
        return False, "invalid_texture"
        
    return True, "valid"

# Input Tabs
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Camera"])
uploaded_file = None

with tab1:
    f = st.file_uploader("Gallery se photo select karein", type=["jpg", "jpeg", "png"])
    if f: uploaded_file = f

with tab2:
    cam = st.camera_input("Darbay ke samne live photo capture karein")
    if cam: uploaded_file = cam

# Prediction Engine
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Sample", use_container_width=True)

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    is_valid, reason = validate_dropping_image(img_bgr)

    if not is_valid:
        if reason == "human_face":
            st.error("🚫 **Insaan / Chehra Detect Hua Hai!**\n\nAI sirf parinday ki beeth (dropping) diagnose karne ke liye trained hai. Meharbani kar ke kisi insaan ya chehre ki tasweer mat daalein.")
        else:
            st.warning("⚠️ **Ghalat Tasweer / Beeth Saaf Nahi Hai!**\n\nZameen par beeth ki kareeb se (close-up) wazeh tasweer khainchein.")
    else:
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

        with st.spinner("AI bimari detect kar raha hai..."):
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            max_idx = np.argmax(preds)
            confidence = float(preds[max_idx]) * 100
            diagnosis = CLASSES[max_idx]

        if confidence < 55.0:
            st.warning("⚠️ **Tasweer Shakki Hai:** Confidence 55% se kam hai. Meharbani kar ke beeth ki behtar roshni mein photo lein.")
        else:
            info = TREATMENTS[diagnosis]

            if diagnosis == "Healthy":
                st.markdown(f"<div class='success-card'><h3>✅ Tashkhees: {info['title']} ({confidence:.1f}%)</h3><p>{info['alert']}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-card'><h3>🚨 Tashkhees: {info['title']} ({confidence:.1f}%)</h3><p><b>Khatra:</b> {info['alert']}</p></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='med-card'>
                    <h4 style='color:#16a34a;'>🌿 Desi Ilaaj</h4>
                    <p>{info['desi']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='med-card'>
                    <h4 style='color:#2563eb;'>💊 Medical Ilaaj</h4>
                    <p>{info['medical']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='med-card'>
                <b>🔍 Alamaat:</b> {info['symptoms']}<br>
                <b>🛡️ Ehtiyaat:</b> {info['care']}
            </div>
            """, unsafe_allow_html=True)