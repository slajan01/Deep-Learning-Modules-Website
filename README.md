# Deep Learning Modules Website

This repository hosts a collection of interactive deep learning modules, each accessible through a simple web interface. The modules demonstrate the practical applications of deep learning in various domains such as natural language processing and computer vision.

## Modules

### 1. Chatbot
- **Description**: A conversational chatbot powered by Hugging Face's `facebook/blenderbot-400M-distill` model. The chatbot can respond to text input and engage in basic conversations.
- **Technology**: Hugging Face Transformers, Flask.
- **How to Use**: Navigate to the Chatbot module, enter your text, and get a response from the chatbot.

### 2. Digit Recognizer
- **Description**: A handwritten digit recognition module powered by a Convolutional Neural Network (CNN). Users can draw a digit on a canvas, and the model predicts the digit.
- **Technology**: TensorFlow, Flask, HTML5 Canvas for drawing interface.
- **How to Use**: Navigate to the Digit Recognizer module, draw a digit on the canvas, and click submit to get the predicted digit.

## Requirements

To run this project locally, ensure you have the following dependencies installed:

- Flask==3.1.0
- keras==3.7.0
- numpy==1.26.4
- opencv-python==4.10.0.84
- pillow==11.1.0
- requests==2.32.3
- tensorflow-cpu==2.18.0
- python-dotenv==1.0.0

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Deployment

The application is deployed on Render. You can access the live website [here](https://deep-learning-modules-website.onrender.com).

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/slajan01/Deep-Learning-Modules-Website.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Deep-Learning-Modules-Website
   ```

3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   flask run
   ```

6. Open your browser and navigate to `http://127.0.0.1:5000/` to interact with the modules.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

## License

This project is licensed under the MIT License.

