# Deep Learning Modules
A Flask-based web app featuring various deep learning modules like chatbots, number classifiers, etc.

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `python app.py`.
4. Access the app at `http://localhost:5000`.

## Chatbot Module

The chatbot module in this project uses the Hugging Face **facebook/blenderbot-400M-distill** model. This is a lightweight conversational model suitable for basic chatbot functionality. The model processes user inputs through the Hugging Face Inference API.

### Key Features:
- Model: `facebook/blenderbot-400M-distill`
- Framework: Hugging Face Transformers
- API: Hugging Face Inference API

For more information on this model, visit its [Hugging Face page](https://huggingface.co/facebook/blenderbot-400M-distill).

