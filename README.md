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

## 📁 Project Structure

```
MNIST_Digit_Recognition/
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
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

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
```bash
python train_model.py
```

This will:
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
