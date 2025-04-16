from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import eventlet
import re

# Load environment variables from .env file
load_dotenv()

# Now you can access the OpenAI API key like this:
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")
app = Flask(__name__)
# socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5000", "http://127.0.0.1:5000"], async_mode='eventlet')
CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"])


current_job_title = "AI/ML Engineer"
conversation_history = []


def generate_job_description(job_title):
    prompt = f"""
    Write a detailed job description for a {job_title}. Include responsibilities, qualifications, and key skills required.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()
    cleaned = re.sub(r"(?i)(Job Title:.*\n?)|(Job Description:.*\n?)", "", raw).strip()

    return cleaned



def chatbot_response_with_history(user_input, job_description=None):
    conversation_history.append({"role": "user", "content": user_input})

    if job_description is None:
        job_description = "the role the user is applying for."

    # Generate the interview-specific system prompt
    prompt = f"""
    You are an AI interviewer conducting a job interview based on this job description:

    {job_description}

    Ask one relevant question at a time based on the user's role, experience, and the job description. 
    After each question, wait for their response and then ask a follow-up. 
    Keep your questions professional, and focus on the job position. 
    Do not list multiple questions at once.
    """ 
    
    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": prompt},
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

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    
    # Process the user input and generate a bot response
    bot_response = chatbot_response_with_history(user_input, current_job_description)
    audio_url = generate_speech(bot_response)  # Generate the speech file and get the URL
    
    return jsonify({
        'response': bot_response,
        'audio': audio_url,
    })

@app.route('/get_job_description', methods=['GET'])
def get_job_description():
    global current_job_description 
    current_job_description = generate_job_description(current_job_title)
    return jsonify({
        'job_title': current_job_title,
        'job_description': current_job_description
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
