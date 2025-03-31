from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import eventlet
import base64
import cv2
import numpy as np
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from fer import FER
# from moviepy.editor import *

# Load environment variables from .env file
load_dotenv()

# Now you can access the OpenAI API key like this:
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")
app = Flask(__name__)
# socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5000", "http://127.0.0.1:5000"], async_mode='eventlet')
CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"])

conversation_history = []

# # Load the pre-trained emotion model (FER-2013 model)
# input_tensor = Input(shape=(48, 48, 1))
# try:
#     emotion_model = load_model('static/models/model.h5', custom_objects={'input_tensor': input_tensor})
#     print("Model loaded successfully")
# except Exception as e:
#     print(f"Error loading model: {e}")


# # Load OpenCV's pre-trained face detector (Haar Cascade)
# face_cascade = cv2.CascadeClassifier('static/models/haarcascade_frontalface_default.xml')


def chatbot_response_with_history(user_input):
    conversation_history.append({"role": "user", "content": user_input})

    # Generate the interview-specific system prompt
    system_prompt = """
    You are an AI interviewer conducting a job interview. Ask the user one relevant question at a time based on their role and experience. 
    After each question, wait for their response and then ask a follow-up question. 
    Keep your questions professional, and focus on the job position. 
    Do not list multiple questions at once.
    """
    
    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_tokens=150,  # Adjust for longer/shorter responses
        temperature=0.7,  # Controls the creativity of the response
    )
    bot_reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": bot_reply})  # Save bot response

    return bot_reply

    
def generate_speech(text):
    """Convert text response to speech using OpenAI TTS."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Available voices: alloy, echo, fable, onyx, nova, shimmer
        input=text
    )
    
    audio_path = "static/audio/response.mp3"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(response.content)

    return audio_path



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/analyze_emotion", methods=["POST"])
def analyze_emotion():
    try:
        # Decode base64 image
        data = request.json["image"]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)

        # Initialize FER detector
        emotions = FER().detect_emotions(img)

        # Find the best emotion with the highest score
        best_emotion = max(
            (max(face['emotions'], key=face['emotions'].get) for face in emotions), 
            key=lambda e: next(face['emotions'][e] for face in emotions if e in face['emotions']),
            default="unknown"
        )

        return jsonify({"emotion": best_emotion})

    except Exception as e:
        print("Emotion detection error:", str(e))
        return jsonify({"emotion": "unknown"})


# from deepface import DeepFace

# @app.route("/analyze_emotion", methods=["POST"])
# def analyze_emotion():
#     try:
#         # Decode base64 image
#         data = request.json["image"]
#         img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)

#         # Use DeepFace for emotion analysis
#         analysis = DeepFace.analyze(img, actions=['emotion'])

#         best_emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
#         return jsonify({"emotion": best_emotion})
    
#     except Exception as e:
#         print("Emotion detection error:", str(e))
#         return jsonify({"emotion": "unknown"})




@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    
    # Process the user input and generate a bot response (similar to your previous logic)
    bot_response = chatbot_response_with_history(user_input)
    audio_url = generate_speech(bot_response)  # Generate the speech file and get the URL
    
    return jsonify({
        'response': bot_response,
        'audio': audio_url
    })

# WebRTC signaling routes using Flask-SocketIO

@socketio.on('offer')
def handle_offer(data):
    print("Received offer:", data)
    # Send the offer to the remote peer
    emit('offer', data, broadcast=True)

@socketio.on('answer')
def handle_answer(data):
    print("Received answer:", data)
    # Send the answer to the remote peer
    emit('answer', data, broadcast=True)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    print("Received ICE candidate:", data)
    # Send the ICE candidate to the remote peer
    emit('ice-candidate', data, broadcast=True)

if __name__ == "__main__":
    os.makedirs("static/audio", exist_ok=True)  # Ensure the audio directory exists
    # app.run(debug=True)
    # socketio.run(app, debug=True)
    socketio.run(app, host='127.0.0.1', port=5000)

