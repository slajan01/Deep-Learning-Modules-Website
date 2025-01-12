# Deep Learning Modules Website

This repository hosts a collection of interactive deep learning modules, deployed as a web application using Flask. Each module demonstrates a specific application of deep learning, accessible via a user-friendly interface.

## Modules Available

### 1. Chatbot Module
- **Description**: An interactive chatbot capable of understanding and responding to user input.
- **Technology**: Utilizes the Hugging Face `facebook/blenderbot-400M-distill` model for conversational AI.
- **Features**:
  - Accepts text input from users.
  - Provides human-like conversational responses.
- **How to Use**:
  - Navigate to the Chatbot Module.
  - Enter your query and click "Submit" to receive a response.

### 2. Digit Recognizer
- **Description**: A digit recognition module that uses a Convolutional Neural Network (CNN) to identify handwritten digits.
- **Technology**: Trained on the MNIST dataset, deployed with TensorFlow.
- **Features**:
  - Interactive canvas for users to draw a digit.
  - Predicts the digit in real-time upon submission.
- **How to Use**:
  - Navigate to the Digit Recognizer.
  - Draw a digit on the canvas and click "Submit" to see the prediction.

### 3. Image Classifier
- **Description**: A module for classifying images into predefined categories using a pre-trained MobileNetV2 model.
- **Technology**: Leverages TensorFlow's MobileNetV2 architecture.
- **Features**:
  - Upload an image to classify its content.
  - Displays the predicted category along with confidence scores.
- **How to Use**:
  - Navigate to the Image Classifier.
  - Upload an image file and click "Submit" to see the classification result.

### 4. Sentiment Analysis
- **Description**: A sentiment analysis module that classifies text as positive, negative, or neutral.
- **Technology**: Uses Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` model.
- **Features**:
  - Input text to analyze its sentiment.
  - Returns the sentiment label (e.g., Positive, Negative, Neutral).
- **How to Use**:
  - Navigate to the Sentiment Analysis Module.
  - Enter text in the provided field and click "Analyze" to see the sentiment.

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/slajan01/Deep-Learning-Modules-Website.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Deep-Learning-Modules-Website
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   flask run
   ```
6. Open the app in your browser at `http://127.0.0.1:5000`.

## Deployment

The application is deployed on [Render](https://render.com). Visit the live application [here](https://deep-learning-modules-website.onrender.com/).

## Future Work
- Add more modules, such as:
  - Object Detection.
  - Text Summarization.
  - Style Transfer.

Feel free to suggest new ideas or contribute to the project!


