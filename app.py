# Importing necessary modules
from flask import Flask, render_template, request, jsonify
import requests

# Initializing Flask application
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Add your Hugging Face API key here
HUGGINGFACE_API_KEY = "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"

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

# Main driver
if __name__ == '__main__':
    app.run(debug=True)