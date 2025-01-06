# Importing necessary modules
import requests
import os
import numpy as np
import base64
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

# Initializing Flask application
app = Flask(__name__)

# Load the model
try:
    cnn_model = load_model('models/digit_recognizer.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image_data):
    # Decode Base64 to a PIL image
    image = Image.open(BytesIO(base64.b64decode(image_data.split(",")[1])))
    image = image.resize((28, 28)).convert('L')  # Resize to 28x28 and convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
    return image

# Route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Add your Hugging Face API key here
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Route for the chatbot module
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('user_input', '')
        
        print(f"User Input: {user_input}")  # Debugging: Print user input

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
                headers=headers,
                json={"inputs": user_input},
            )

            print(f"API Response: {response.json()}")  # Debugging: Print API response

            if response.status_code == 200:
                bot_response = response.json()[0].get('generated_text', 'No response generated.')
            else:
                bot_response = f"Error {response.status_code}: {response.text}"

        except Exception as e:
            print(f"Exception occurred: {e}")
            bot_response = "Sorry, an error occurred while processing your input."

        return jsonify({'response': bot_response})
    return render_template('chatbot.html')

@app.route('/digit-recognizer', methods=['GET', 'POST'])
def digit_recognizer():
    if request.method == 'POST':
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Preprocess the image and predict
        try:
            processed_image = preprocess_image(image_data)
            print(f"Preprocessed image shape: {processed_image.shape}")
            prediction = cnn_model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            print(f"Model prediction: {prediction}, Predicted digit: {predicted_digit}")
            return jsonify({'digit': int(predicted_digit)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('digit_recognizer.html')

# Main driver
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)