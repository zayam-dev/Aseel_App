import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Aseel AI - Health Diagnostic",
    page_icon="🐓",
    layout="centered"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        background: #ffffff;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .header-text h2 { margin: 0; color: #0f172a; font-size: 24px; font-weight: 700; }
    .header-text p { margin: 4px 0 0 0; color: #64748b; font-size: 13px; }
    .alert-danger {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 14px;
        border-radius: 8px;
        margin: 14px 0;
        color: #9f1239;
    }
    .alert-success {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 14px;
        border-radius: 8px;
        margin: 14px 0;
        color: #14532d;
    }
    .treatment-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# 1. Logo Loader (Linux Case-Sensitivity Safe)
logo_file = None
for fname in ["ASEEL.PNG", "ASEEL.png", "aseel.png", "aseel.PNG"]:
    target = os.path.join(os.path.dirname(__file__), fname)
    if os.path.exists(target):
        logo_file = target
        break

# Header UI
col_a, col_b = st.columns([1, 4])
with col_a:
    if logo_file:
        st.image(logo_file, width=85)
    else:
        st.markdown("<h1 style='margin:0;'>🐓</h1>", unsafe_allow_html=True)

with col_b:
    st.markdown("""
        <div style='padding-top: 5px;'>
            <h2 style='margin:0; font-size:24px; color:#1e293b;'>Aseel Disease Diagnostic AI</h2>
            <p style='margin:4px 0 0 0; color:#64748b; font-size:13px;'>Bilingual Poultry Disease Detection & Treatment Guide</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# 2. Model Loader (Only Caches TFLite Interpreter)
@st.cache_resource
def load_tflite_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ['Coccidiosis', 'Healthy', 'New Castle Disease (Rani Khet)', 'Salmonella']

TREATMENTS = {
    'Coccidiosis': {
        'title': 'Khuni / Surkh Beeth (Coccidiosis)',
        'alert': 'Murghay ko fauri alag darbay mein band karein aur zameen bilkul khushk rakhein.',
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
        'symptoms': 'Bandhi hui theek beeth, koi bemari ki alamat nahi.',
        'care': 'Rozana taza paani aur saaf daana jari rakhein.',
        'desi': 'Kisi desi nuskhe ki zaroorat nahi.',
        'medical': 'Routine multivitamins paani mein dein.'
    }
}

# 3. Crash-Proof Image Validator (Checks Faces & Non-Dropping Photos)
def validate_image_content(img_bgr):
    # Safe Cascade Face Check
    try:
        xml_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        if os.path.exists(xml_path):
            face_cascade = cv2.CascadeClassifier(xml_path)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                return False, "human_face"
    except Exception:
        pass

    # Pure-NumPy Skin Color Ratio Check (Catch selfies without crashing)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 175) & (cb >= 80) & (cb <= 130)
    skin_ratio = np.mean(skin_mask)
    if skin_ratio > 0.22:
        return False, "human_face"

    # Texture & Contrast Check
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brightness_variance = np.var(hsv[:, :, 2])
    unique_colors = len(np.unique(hsv[:, :, 0]))
    if brightness_variance < 25 or unique_colors < 12:
        return False, "invalid_texture"

    return True, "valid"

# Input Interface
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Camera"])
uploaded_file = None

with tab1:
    f = st.file_uploader("Gallery se beeth ki photo chunein", type=["jpg", "jpeg", "png"])
    if f: uploaded_file = f

with tab2:
    cam = st.camera_input("Darbay ke samne live photo capture karein")
    if cam: uploaded_file = cam

# Prediction Engine
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    is_valid, reason = validate_image_content(img_bgr)

    if not is_valid:
        if reason == "human_face":
            st.error("🚫 **Insaan / Chehra Detect Hua Hai!**\n\nAI sirf parindon ki beeth (droppings) ko diagnose karta hai. Apni ya kisi insaan ki photo na dalein.")
        else:
            st.warning("⚠️ **Ghalat Tasweer:** Beeth saaf nazar nahi aa rahi. Zameen par beeth ki close-up tasweer upload karein.")
    else:
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

        with st.spinner("AI photo ko analyze kar raha hai..."):
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            max_idx = np.argmax(preds)
            confidence = float(preds[max_idx]) * 100
            diagnosis = CLASSES[max_idx]

        # Out-of-domain safe threshold
        if confidence < 60.0:
            st.warning(f"⚠️ **Ghair-Wazeh Nateeja ({confidence:.1f}%):** AI is photo ko kisi bimari se match nahi kar pa raha. Kripya saaf roshni mein close-up tasweer lein.")
        else:
            info = TREATMENTS[diagnosis]

            if diagnosis == "Healthy":
                st.markdown(f"""
                <div class='alert-success'>
                    <h3 style='margin:0;'>✅ Nateeja: {info['title']} ({confidence:.1f}%)</h3>
                    <p style='margin:6px 0 0 0;'>{info['alert']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='alert-danger'>
                    <h3 style='margin:0;'>🚨 Nateeja: {info['title']} ({confidence:.1f}%)</h3>
                    <p style='margin:6px 0 0 0;'><b>Khatra:</b> {info['alert']}</p>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='treatment-card'>
                    <h4 style='color:#16a34a; margin:0 0 8px 0;'>🌿 Desi Ilaaj</h4>
                    <p style='margin:0; font-size:14px; color:#334155;'>{info['desi']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='treatment-card'>
                    <h4 style='color:#2563eb; margin:0 0 8px 0;'>💊 Medical Ilaaj</h4>
                    <p style='margin:0; font-size:14px; color:#334155;'>{info['medical']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='treatment-card'>
                <p style='margin:0 0 6px 0;'><b>🔍 Alamaat:</b> {info['symptoms']}</p>
                <p style='margin:0;'><b>🛡️ Ehtiyaat:</b> {info['care']}</p>
            </div>
            """, unsafe_allow_html=True)