# MNIST Digit Recognition

A web application that uses a trained Convolutional Neural Network (CNN) to recognize handwritten digits (0-9).

## Features
- Draw digits on a canvas
- Real-time prediction using TensorFlow/Keras model
- Responsive Bootstrap UI

## Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model (if needed)
python train_model.py

# Run app
python app.py
```

Visit http://127.0.0.1:5000

## Hugging Face Deployment

This app is deployed at: `https://YOUR_USERNAME-mnist-digit-recognition.hf.space`

### Deploy Your Own
1. Create a [Hugging Face Space](https://huggingface.co/spaces) with Docker SDK
2. Upload all files from this repository
3. The app will auto-deploy

## Tech Stack
- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML5 Canvas, JavaScript, Bootstrap
- **Model**: CNN trained on MNIST dataset
