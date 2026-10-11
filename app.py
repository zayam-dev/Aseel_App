import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# TFLite Model Load (Uses very low RAM)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ['Coccidiosis', 'Healthy', 'New Castle Disease (Rani Khet)', 'Salmonella']

TREATMENTS = {
    'Coccidiosis': {
        'class_key': 'coccidiosis',
        'disease_en': 'Coccidiosis (Bloody / Red Dropping)',
        'disease_roman': 'Khuni / Surkh Beeth (Coccidiosis)',
        'alert_en': 'Quarantine bird immediately. Keep floor dry.',
        'alert_roman': 'Murghay ko fauri alag darbay mein band karein aur zameen khushk rakhein.',
        'symptoms_en': 'Bloody reddish droppings, pale face, drooping wings.',
        'symptoms_roman': 'Surkh ya khoon aalood beeth, chehray ka peela parna, par latkana.',
        'care_en': 'Prevent wet bedding and do not feed cold water.',
        'care_roman': 'Geeli mitti ya gandi jagah se door rakhein, thanda paani mat dein.',
        'desi_en': 'Apple cider vinegar (5ml/L) and garlic extract in water.',
        'desi_roman': 'Saib ka sirka (5ml fi liter) aur lehsan ka paani dein.',
        'medical_en': 'Vet Amprolium powder or Baycox (1ml per liter water for 3-5 days).',
        'medical_roman': 'Amprolium powder ya Baycox (1ml fi liter paani mein 3-5 din).'
    },
    'New Castle Disease (Rani Khet)': {
        'class_key': 'ranikhet',
        'disease_en': 'Newcastle Disease / Rani Khet (Greenish Dropping)',
        'disease_roman': 'Sabz Beeth / Lakwa Jhatka (Rani Khet)',
        'alert_en': 'Critical virus! Isolate immediately to prevent transmission.',
        'alert_roman': 'Shadeed khatarnak bimari! Doosre parindon se fauri door karein.',
        'symptoms_en': 'Greenish watery droppings, neck twisting, tremors.',
        'symptoms_roman': 'Sabz patli beeth, gardan ka ghumna, jhatkay lagna.',
        'care_en': 'Keep room warm, disinfect cage thoroughly.',
        'care_roman': 'Darbay ko garam rakhein aur spray se jaraseem-kash karein.',
        'desi_en': 'Ajwain, gur (jaggery) and warm ginger decoction.',
        'desi_roman': 'Gur, sonf aur ajwain ka garam qehwa bana kar dein.',
        'medical_en': 'Neurobion syrup (vitamins) + broad-spectrum antibiotic (Enrofloxacin).',
        'medical_roman': 'Neurobion drops taqat ke liye aur Enrofloxacin antibiotic dein.'
    },
    'Salmonella': {
        'class_key': 'salmonella',
        'disease_en': 'Salmonella / Pullorum (Chalky White Dropping)',
        'disease_roman': 'Safaid / Chuna Beeth (Salmonella)',
        'alert_en': 'Bacterial infection. Disinfect all feed and water drinkers.',
        'alert_roman': 'Bartan saaf karein aur taza ubla hua paani dein.',
        'symptoms_en': 'Chalky white paste around vent, loss of appetite, fever.',
        'symptoms_roman': 'Peechay safaid chuna jaisi beeth ka lagna, bhook na lagna.',
        'care_en': 'Avoid stale grain or contaminated drinkers.',
        'care_roman': 'Purana daana ya kharab paani hargiz na dein.',
        'desi_en': 'Turmeric (Haldi) mixed in boiled egg yolk.',
        'desi_roman': 'Ublay anday ki zardi mein haldi mila kar dein.',
        'medical_en': 'Neomycin or Gentamicin drops / Oxytetracycline powder.',
        'medical_roman': 'Gentamicin injection ya Oxytetracycline powder 4-5 din dein.'
    },
    'Healthy': {
        'class_key': 'healthy',
        'disease_en': 'Healthy Dropping (Normal)',
        'disease_roman': 'Sehatmand Parinda (Normal)',
        'alert_en': 'No disease detected. Continue normal hygiene.',
        'alert_roman': 'MashaAllah parinda bilkul theek hai.',
        'symptoms_en': 'Firm droppings with normal white capping.',
        'symptoms_roman': 'Bandhi hui theek beeth.',
        'care_en': 'Maintain balanced diet and fresh water daily.',
        'care_roman': 'Rozana taza paani aur achi khurak jari rakhein.',
        'desi_en': 'N/A',
        'desi_roman': 'N/A',
        'medical_en': 'N/A',
        'medical_roman': 'N/A'
    }
}

INVALID_RESPONSE = {
    'class_key': 'invalid',
    'disease_en': 'Invalid Photo / Not a Dropping',
    'disease_roman': 'Ghalat Tasweer / Beeth Nazar Nahi Aayi',
    'alert_en': 'Please upload a clear close-up picture of the dropping on floor.',
    'alert_roman': 'Meharbani farma kar sirf beeth ki saaf photo khinch kar upload karein.',
    'symptoms_en': 'N/A',
    'symptoms_roman': 'N/A',
    'care_en': 'N/A',
    'care_roman': 'N/A',
    'desi_en': 'N/A',
    'desi_roman': 'N/A',
    'medical_en': 'N/A',
    'medical_roman': 'N/A',
    'confidence': 0.0
}

def is_valid_dropping(image_cv):
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    brightness_variance = np.var(v_channel)
    h_channel = hsv[:, :, 0]
    unique_colors = len(np.unique(h_channel))
    return brightness_variance >= 30 and unique_colors >= 15

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/logo')
def get_logo():
    return send_from_directory('.', 'ASEEL.png')

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    results = []

    for file in files:
        if file.filename == '':
            continue

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_cv is None or not is_valid_dropping(img_cv):
            results.append(INVALID_RESPONSE)
            continue

        # Prepare for TFLite inference
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        max_idx = np.argmax(predictions)
        confidence = float(predictions[max_idx]) * 100
        predicted_class = CLASSES[max_idx]

        if confidence < 50.0:
            results.append(INVALID_RESPONSE)
            continue

        data = TREATMENTS[predicted_class].copy()
        data['confidence'] = round(confidence, 1)
        results.append(data)

    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)