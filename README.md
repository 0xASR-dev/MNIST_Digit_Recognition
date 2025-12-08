<<<<<<< HEAD
# MNIST Digit Recognition 🔢

A web application that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) drawn on an HTML canvas.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)

## 🎯 Features

- **Interactive Drawing Canvas** - Draw digits with your mouse
- **Real-time Predictions** - Get instant digit recognition
- **High Accuracy** - CNN model trained to 99%+ accuracy on MNIST dataset
- **Confidence Score** - See how confident the model is in its prediction

## 🖥️ Demo

Draw a digit on the canvas and click "Predict" to see the model's prediction!
=======
# MNIST Digit Recognition Web Application

A deep learning-powered web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Draw a digit on the canvas and watch the AI predict it in real-time!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Interactive Drawing Canvas**: Draw digits directly in your browser with a smooth, responsive canvas
- **Real-Time Predictions**: Get instant predictions powered by a trained CNN model
- **High Accuracy**: Model achieves high accuracy on the MNIST test dataset through data augmentation and dropout regularization
- **Modern UI**: Clean, responsive interface built with Bootstrap 5
- **Easy Deployment**: Simple Flask-based architecture for easy hosting

## 🚀 Technologies Used

- **Backend**:
  - Python 3.8+
  - Flask 3.1.2 - Web framework
  - TensorFlow/Keras 2.20.0 - Deep learning framework
  - NumPy 2.3.2 - Numerical computing
  - Pillow 11.3.0 - Image processing

- **Frontend**:
  - HTML5 Canvas for drawing
  - Bootstrap 5 for responsive design
  - Vanilla JavaScript for interactivity

- **Machine Learning**:
  - Convolutional Neural Network (CNN)
  - Data augmentation for improved generalization
  - Dropout layers for regularization
>>>>>>> d2352feccc0bd42a72ffbcab0c6cc4e39345bdae

## 📁 Project Structure

```
MNIST_Digit_Recognition/
<<<<<<< HEAD
├── app.py                 # Flask application with prediction endpoint
├── train_model.py         # CNN model training script
├── mnist_model.keras      # Trained Keras model
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration for deployment
├── templates/
│   └── index.html         # Frontend HTML
└── static/
    └── js/
        └── main.js        # Canvas drawing and API logic
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MNIST_Digit_Recognition.git
   cd MNIST_Digit_Recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
=======
│
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── mnist_model.keras       # Pre-trained model file
├── requirements.txt        # Python dependencies
│
├── templates/
│   └── index.html         # Main web interface
│
├── static/
│   └── js/
│       └── main.js        # Canvas drawing and prediction logic
│
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/0xASR-dev/MNIST_Digit_Recognition.git
   cd MNIST_Digit_Recognition
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
>>>>>>> d2352feccc0bd42a72ffbcab0c6cc4e39345bdae
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
4. **Run the application**
=======
## 🎯 Usage

### Running the Web Application

1. **Start the Flask server**
>>>>>>> d2352feccc0bd42a72ffbcab0c6cc4e39345bdae
   ```bash
   python app.py
   ```

<<<<<<< HEAD
5. **Open in browser**
=======
2. **Open your browser** and navigate to:
>>>>>>> d2352feccc0bd42a72ffbcab0c6cc4e39345bdae
   ```
   http://127.0.0.1:5000
   ```

<<<<<<< HEAD
## 🧠 Model Architecture

The CNN model consists of:

| Layer | Description |
|-------|-------------|
| Conv2D (32 filters) | 3x3 kernel, ReLU activation |
| MaxPooling2D | 2x2 pool size |
| Dropout (0.25) | Regularization |
| Conv2D (64 filters) | 3x3 kernel, ReLU activation |
| MaxPooling2D | 2x2 pool size |
| Dropout (0.25) | Regularization |
| Flatten | Converts to 1D |
| Dense (128) | Fully connected, ReLU |
| Dropout (0.5) | Regularization |
| Dense (10) | Output layer, Softmax |

### Training

To retrain the model:
=======
3. **Draw a digit** (0-9) on the canvas using your mouse

4. **Click "Predict"** to see the AI's prediction

5. **Click "Clear"** to reset the canvas and draw again

### Training the Model (Optional)

If you want to retrain the model with different parameters:

>>>>>>> d2352feccc0bd42a72ffbcab0c6cc4e39345bdae
```bash
python train_model.py
```

This will:
<<<<<<< HEAD
- Download the MNIST dataset
- Train for 15 epochs with data augmentation
- Save the model as `mnist_model.keras`

## 🐳 Docker Deployment

### Build and run locally
```bash
docker build -t mnist-recognition .
docker run -p 7860:7860 mnist-recognition
```

### Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Upload these files:
   - `Dockerfile`
   - `app.py`
   - `requirements.txt`
   - `mnist_model.keras`
   - `templates/index.html`
   - `static/js/main.js`

## 📊 API Endpoint

### POST `/predict`

**Request:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "digit": 7,
  "confidence": 98.5
}
```

## 🛠️ Tech Stack

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML5 Canvas, JavaScript, Bootstrap 5
- **Deployment**: Docker, Gunicorn

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
=======
- Download the MNIST dataset (if not cached)
- Train a CNN model with data augmentation
- Save the trained model as `mnist_model.keras`

**Training parameters:**
- Epochs: 15
- Batch size: 128
- Optimizer: Adam
- Data augmentation: rotation, zoom, width/height shifts

## 🧠 Model Architecture

The CNN model consists of the following layers:

```
Conv2D(32, 3x3, relu) → MaxPooling(2x2) → Dropout(0.25)
    ↓
Conv2D(64, 3x3, relu) → MaxPooling(2x2) → Dropout(0.25)
    ↓
Flatten() → Dense(128, relu) → Dropout(0.5)
    ↓
Dense(10, softmax) [Output Layer]
```

**Key Features:**
- **Two convolutional layers** for feature extraction
- **MaxPooling layers** for dimensionality reduction
- **Dropout layers** (0.25 and 0.5) to prevent overfitting
- **Data augmentation** during training (rotation, zoom, shifts)
- **Softmax activation** for multi-class classification (10 digits)

## 📊 Model Performance

The model achieves high accuracy on the MNIST test dataset through:
- Convolutional layers for spatial feature learning
- Data augmentation for better generalization
- Dropout regularization to prevent overfitting
- 15 epochs of training with the Adam optimizer

## 🖼️ How It Works

### Workflow

1. **User draws a digit** on the HTML5 canvas
2. **Canvas converts** the drawing to a PNG image (280x280 pixels)
3. **Frontend sends** the base64-encoded image to the Flask backend
4. **Backend preprocesses** the image:
   - Decodes base64 data
   - Converts to grayscale
   - Resizes to 28x28 pixels
   - Inverts colors (white background → black, black digit → white)
   - Normalizes pixel values to [0, 1]
5. **Model predicts** the digit using the trained CNN
6. **Result is sent** back to the frontend and displayed

### Image Preprocessing Pipeline

```python
Base64 Image → PIL Image → Grayscale → Resize(28x28) 
→ Invert Colors → Normalize → Reshape(1, 28, 28, 1) → Predict
```

## 🎨 Screenshots

### Web Interface
The application features a clean, intuitive interface where users can:
- Draw digits using their mouse or touchpad
- See predictions instantly
- Clear the canvas to try again

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Ideas for Contributions

- Add touch support for mobile devices
- Implement confidence scores for predictions
- Add visualization of the model's internal layers
- Support for custom model training parameters via UI
- Add ability to save and share drawings
- Implement batch prediction for multiple digits

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Created by Yann LeCun and Corinna Cortes
- **TensorFlow/Keras**: For providing excellent deep learning tools
- **Flask**: For the lightweight and flexible web framework
- **Bootstrap**: For the responsive UI components

## 📧 Contact

For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/0xASR-dev/MNIST_Digit_Recognition).

---

**Made with ❤️ using TensorFlow and Flask**
>>>>>>> d2352feccc0bd42a72ffbcab0c6cc4e39345bdae
