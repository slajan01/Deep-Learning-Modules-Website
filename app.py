# Importing necessary modules
import requests
import os
import numpy as np
import base64
import tensorflow as tf
import cv2
from transformers import pipeline
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
from io import BytesIO

# Initializing Flask application
app = Flask(__name__)

# Load the model for digit recognizer
try:
    cnn_model = load_model('models/digit_recognizer.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Load pre-trained model for image classification
model = MobileNetV2(weights="imagenet")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="prajjwal1/bert-mini", from_pt=True
)

# CUDA related warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def preprocess_image(image_data):

    # Decode Base64 to a PIL image
    image = Image.open(BytesIO(base64.b64decode(image_data.split(",")[1])))

    # Ensure the transparent background is replaced with black
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))  # Black background
    image = Image.alpha_composite(background, image)

    # Convert to grayscale
    image = image.convert('L')

    # Convert to a NumPy array for explicit inversion
    image_array = np.array(image)

    # Check if the image is already white on black
    mean_pixel_value = np.mean(image_array)
    if mean_pixel_value > 127:  # Image is black on white
        image_array = 255 - image_array  # Explicit inversion to white on black

    # Convert back to PIL Image after inversion
    image = Image.fromarray(image_array)

    # Find the bounding box of the digit
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize the cropped digit to fit 20x20
    image = image.resize((20, 20), Image.Resampling.LANCZOS)

    # Create a new 28x28 black image and paste the resized digit in the center
    new_image = Image.new('L', (28, 28), 0)  # Black background
    new_image.paste(image, (4, 4))  # Center the digit

    # Normalize pixel values to [0, 1]
    image_array = np.array(new_image).astype('float32') / 255.0

    # Reshape to (1, 28, 28, 1) for the model
    image_array = np.expand_dims(image_array, axis=(0, -1))

    return image_array

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

# Route for the digit recognizer module
@app.route('/digit-recognizer', methods=['GET', 'POST'])
def digit_recognizer():
    if request.method == 'GET':
        # Render the HTML template for the digit recognizer page
        return render_template('digit_recognizer.html')

    if request.method == 'POST':
        # Handle image data for prediction
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        try:
            # Preprocess the image
            processed_image = preprocess_image(image_data)
            print(f"Preprocessed image shape: {processed_image.shape}")
            print(f"Preprocessed image values (sample): {processed_image[0, :, :, 0]}")

            # Predict using the model
            prediction = cnn_model.predict(processed_image)
            predicted_digit = int(np.argmax(prediction))
            print(f"Model prediction: {prediction}, Predicted digit: {predicted_digit}")

            return jsonify({'digit': predicted_digit})
        except Exception as e:
            import traceback
            traceback.print_exc()  # Log the full traceback for debugging
            return jsonify({'error': str(e)}), 500

# Route for the image classifier module
@app.route('/image-classifier', methods=['GET', 'POST'])
def image_classifier():
    if request.method == 'GET':
        return render_template('image_classifier.html')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the file to a temporary location
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using the model
        predictions = model.predict(img_array)
        label = decode_predictions(predictions, top=1)[0][0][1]
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({'label': label})
    
# Route for the sentiment analysis module
@app.route('/sentiment-analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'GET':
        return render_template('sentiment_analysis.html')

    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        try:
            # Analyze sentiment
            results = sentiment_pipeline(text)
            sentiment = results[0]['label']  # Extract the sentiment label
            return jsonify({'sentiment': sentiment})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
# Main driver
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)