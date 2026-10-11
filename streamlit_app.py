import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Aseel Disease Detector", page_icon="🐓", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ['Coccidiosis', 'Healthy', 'New Castle Disease (Rani Khet)', 'Salmonella']

TREATMENTS = {
    'Coccidiosis': {
        'title': 'Khuni / Surkh Beeth (Coccidiosis)',
        'alert': 'Murghay ko fauri alag darbay mein band karein aur zameen khushk rakhein.',
        'symptoms': 'Surkh ya khoon aalood beeth, chehray ka peela parna, par latkana.',
        'desi': 'Saib ka sirka (5ml fi liter) aur lehsan ka paani dein.',
        'medical': 'Vet Amprolium powder ya Baycox (1ml fi liter paani mein 3-5 din).'
    },
    'New Castle Disease (Rani Khet)': {
        'title': 'Sabz Beeth / Lakwa Jhatka (Rani Khet)',
        'alert': 'Shadeed khatarnak virus! Doosre parindon se fauri door karein.',
        'symptoms': 'Sabz patli beeth, gardan ka ghumna, jhatkay lagna.',
        'desi': 'Gur, sonf aur ajwain ka garam qehwa bana kar dein.',
        'medical': 'Neurobion drops taqat ke liye aur Enrofloxacin antibiotic dein.'
    },
    'Salmonella': {
        'title': 'Safaid / Chuna Beeth (Salmonella)',
        'alert': 'Bartan saaf karein aur taza ubla hua paani dein.',
        'symptoms': 'Peechay safaid chuna jaisi beeth ka lagna, bhook na lagna.',
        'desi': 'Ublay anday ki zardi mein haldi mila kar dein.',
        'medical': 'Gentamicin drops ya Oxytetracycline powder 4-5 din dein.'
    },
    'Healthy': {
        'title': 'Sehatmand Parinda (Healthy)',
        'alert': 'MashaAllah parinda bilkul theek hai.',
        'symptoms': 'Bandhi hui theek beeth.',
        'desi': 'N/A',
        'medical': 'Rozana taza paani aur achi khurak jari rakhein.'
    }
}

st.title("🐓 Aseel Disease Diagnostic AI")
st.write("Upload dropping image ya live camera se scan karein:")

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Camera"])

uploaded_file = None
with tab1:
    file = st.file_uploader("Gallery se photo choose karein", type=["jpg", "jpeg", "png"])
    if file:
        uploaded_file = file

with tab2:
    cam = st.camera_input("Darbay ke samne live photo lein")
    if cam:
        uploaded_file = cam

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Photo", use_container_width=True)

    # Preprocessing
    img_cv = np.array(image)
    img_resized = cv2.resize(img_cv, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

    with st.spinner("AI analyzing dropping..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        max_idx = np.argmax(preds)
        confidence = float(preds[max_idx]) * 100
        diagnosis = CLASSES[max_idx]

    st.divider()

    if confidence < 50.0:
        st.error("⚠️ Invalid Photo: Beeth saaf nazar nahi aayi. Meharbani kar ke close-up photo lein.")
    else:
        info = TREATMENTS[diagnosis]
        if diagnosis == "Healthy":
            st.success(f"### Result: {info['title']} ({confidence:.1f}%)")
        else:
            st.error(f"### Result: {info['title']} ({confidence:.1f}%)")
            st.warning(f"**Alert:** {info['alert']}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌿 Desi Ilaaj")
            st.info(info['desi'])
        with col2:
            st.markdown("#### 💊 Medical Ilaaj")
            st.info(info['medical'])

        st.markdown(f"**Symptoms:** {info['symptoms']}")