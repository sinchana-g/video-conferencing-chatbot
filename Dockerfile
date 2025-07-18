FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget unzip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Vosk model
COPY models/ models/

# Copy app code
COPY . .

# Expose Flask port
EXPOSE 8080

CMD ["python", "app.py"]
