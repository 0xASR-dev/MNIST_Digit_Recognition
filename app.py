# app.py
# This script creates a Flask web application for MNIST digit recognition.

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_model.keras')

def preprocess_image(image_data):
    """
    Preprocesses the image from the canvas to be compatible with the model.
    """
    # Decode the base64 image
    img_str = image_data.split(',')[1]
    img_bytes = base64.b64decode(img_str)
    
    # Open the image using PIL
    img = Image.open(io.BytesIO(img_bytes))

    # Create a new image with a white background
    # The original image from canvas is RGBA with a transparent background
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3]) # 3 is the alpha channel

    # Convert to grayscale
    img = bg.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert colors (the model was trained on black background, white digit)
    img_array = 255 - img_array
    
    # Reshape and normalize
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    
    # Preprocess the image
    processed_image = preprocess_image(image_data)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    predicted_digit = int(np.argmax(prediction))
    
    return jsonify({'digit': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
