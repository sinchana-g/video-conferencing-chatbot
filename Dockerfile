# Use a Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files into the container
COPY . .

# Expose the port that Flask app will run on
EXPOSE 8080

# Run the Flask app using Gunicorn (production-ready WSGI server)
#CMD ["gunicorn", "-k", "eventlet", "-b", "0.0.0.0:8080", "app:app"]

CMD ["python", "app.py"]
