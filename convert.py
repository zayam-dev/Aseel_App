import tensorflow as tf

print("Loading Keras model...")
model = tf.keras.models.load_model('best_aseel_model_v2.keras')

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! 'model.tflite' ban gaya hai.")