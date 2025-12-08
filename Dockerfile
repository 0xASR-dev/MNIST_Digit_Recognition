# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY mnist_model.keras .
COPY templates/ templates/
COPY static/ static/

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
