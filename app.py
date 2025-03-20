from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can access the OpenAI API key like this:
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")
app = Flask(__name__)
CORS(app, origins="http://127.0.0.1:5000")

conversation_history = []

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
        model="gpt-4",  # Or use "gpt-3.5-turbo" for the earlier model
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

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

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
    app.run(debug=True)
    # socketio.run(app, debug=True, port=5000)

