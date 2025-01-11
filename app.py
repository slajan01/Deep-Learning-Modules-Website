# Importing necessary modules
import requests
import os
import numpy as np
import base64
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
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

    # Ensure the transparent background is replaced with black
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))  # Black background
    image = Image.alpha_composite(background, image)

    # Convert to grayscale
    image = image.convert('L')
    print(f"After grayscale conversion: {image.mode}")  # Debug: Check mode

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Explicitly invert the pixel values
    image_array = 255 - image_array
    print(f"Pixel values after explicit inversion (sample): {image_array[:5, :5]}")

    # Convert back to PIL Image for further processing
    image = Image.fromarray(image_array)

    # Save the explicitly inverted image for debugging
    image.save("debug_explicitly_inverted_image.png")

    # Find the bounding box of the digit
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize and center the digit
    image = image.resize((20, 20), Image.Resampling.LANCZOS)
    new_image = Image.new('L', (28, 28), 0)
    new_image.paste(image, (4, 4))

    # Save the processed image for debugging
    new_image.save("debug_final_image_after_explicit_inversion.png")
    print("Processed image saved for debugging.")

    # Normalize the image
    image_array = np.array(new_image).astype('float32') / 255.0
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

# Main driver
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)