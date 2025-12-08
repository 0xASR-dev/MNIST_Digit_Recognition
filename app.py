# This script creates a Flask application for MNIST digit recognition.

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import io
import os
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
model = tf.keras.models.load_model('mnist_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data['image']
        
        # Remove the data URL prefix (e.g., "data:image/png;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Handle PNG with alpha channel - composite onto white background
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Invert colors (canvas has black on white, MNIST expects white on black)
        image_array = 255 - image_array
        
        # Find bounding box of the digit (non-zero pixels)
        rows = np.any(image_array > 20, axis=1)
        cols = np.any(image_array > 20, axis=0)
        
        if not rows.any() or not cols.any():
            return jsonify({'digit': 0, 'confidence': 0})
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop to bounding box
        cropped = image_array[rmin:rmax+1, cmin:cmax+1]
        
        # Add padding to make it square with some margin
        h, w = cropped.shape
        max_dim = max(h, w)
        
        # Create square canvas with padding (20% margin on each side)
        pad_size = int(max_dim * 0.2)
        new_size = max_dim + 2 * pad_size
        
        square_img = np.zeros((new_size, new_size), dtype=np.uint8)
        
        # Center the digit in the square
        y_offset = (new_size - h) // 2
        x_offset = (new_size - w) // 2
        square_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        
        # Resize to 28x28 (MNIST input size)
        pil_img = Image.fromarray(square_img)
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize to [0, 1]
        final_array = np.array(pil_img).astype('float32') / 255.0
        
        # Reshape for the model (batch_size, height, width, channels)
        final_array = final_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        predictions = model.predict(final_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': round(confidence * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

