import tensorflow as tf
import re
import string
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
import requests
from flask import Flask,request,jsonify

app = Flask(__name__)

# Emotion mapping for keywords
KEYWORD_EMOTION_MAP = {
    'stress': 'stress',
    'anxiety': 'stress',
    'happy': 'happy',
    'joy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'disgust': 'angry'
}

# Emotion mapping from predictions
def map_emotions(emotion):
    if emotion == 'fear':
        return 'stress'
    elif emotion == 'joy':
        return 'happy'
    elif emotion == 'sadness':
        return 'sad'
    elif emotion == 'anger':
        return 'angry'
    else:
        return 'neutral'

# Detect emotion from keywords
def detect_emotion_from_keywords(text):
    for keyword, emotion in KEYWORD_EMOTION_MAP.items():
        if keyword.lower() in text.lower():
            print(f"Keyword emotion found: {emotion}")
            return emotion
    print("No keyword emotion found")
    return None

# Load the saved model
model = tf.keras.models.load_model('moodAnalysis.h5')
print("Model successfully loaded. Summary:")
model.summary()

# Load tokenizer and label encoder
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open('labelEncoder.pickle', 'rb') as file:
    label_encoder = pickle.load(file)

# Clean text
def clean(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

# Preprocess text for model input
def preprocess_text(text, tokenizer, max_len=256):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, truncating='pre')
    return padded

# Predict emotion
def predict_emotion(text):
    cleaned_text = clean(text)
    input_vector = preprocess_text(cleaned_text, tokenizer)
    prediction = model.predict(input_vector)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    print(f"Predicted label: {predicted_label}")
    return predicted_label

def emotion_detection(text):
  print(f"Received text: {text}")
  # Check for keyword-based emotion detection
  keyword_emotion = detect_emotion_from_keywords(text)
  if keyword_emotion:
    print(f"Keyword-Based Emotion Detected: {keyword_emotion}")
    return keyword_emotion
    
  # If no keyword-based emotion, use model prediction
  model_emotion = predict_emotion(text)
  classified_emotion = map_emotions(model_emotion)
  print(f"Model-Based Emotion Detected: {classified_emotion}")
  return classified_emotion

print(emotion_detection("I think I am very ok with it"))

@app.route('/',methods=['POST'])
def predict():
    text = request.json['text']
    print(text)
    result = emotion_detection
    return jsonify({"emotion":result})
    
if __name__ == "__main__":
    app.run(debug=True)
