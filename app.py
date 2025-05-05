from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import eventlet
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# Load environment variables from .env file
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app, origins="*")

scenarios_df = pd.read_excel("Cleaned_Scenarios_by_Skill.xlsx")
scenarios_df.columns = scenarios_df.columns.str.strip()
unique_skills = scenarios_df["Skill Name"].dropna().unique().tolist()
skill_embeddings = client.embeddings.create(
    model="text-embedding-ada-002",
    input=unique_skills
).data
skill_embedding_matrix = normalize(np.array([e.embedding for e in skill_embeddings]))

current_job_title = "Construction Worker"
conversation_history = []
scenario_history = []


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


def extract_skills_from_jd(jd_text):
    prompt = f"""Extract the top 5 most relevant soft/behavioral skills from this job description:
    
    {jd_text}
    
    List them one per line concise phrases (2–3 words max). 
    For example: Communication, Teamwork, Problem Solving, Adaptability, Leadership."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]


def find_top_matching_skills(jd_skills, top_n=2):
    # Embed the job description skills
    response = client.embeddings.create(model="text-embedding-ada-002", input=jd_skills)
    jd_embeddings = normalize(np.array([e.embedding for e in response.data]))

    # Calculate cosine similarities with the skill embedding matrix for each job description skill
    skill_similarities = cosine_similarity(jd_embeddings, skill_embedding_matrix)
    
    # Sum up the cosine similarities for each job description skill
    avg_similarities = skill_similarities.mean(axis=0)
    
    # Get top N indices based on the similarity score
    top_indices = np.argsort(avg_similarities)[::-1]
    
    # Extract the top skills (with duplicates initially allowed)
    top_skills = [unique_skills[i] for i in top_indices]
    
    # Normalize skill matching (lowercase and strip spaces)
    normalized_skills = [skill.strip().lower() for skill in top_skills]
    
    # Remove duplicates by converting the list to a set and back to a list
    unique_normalized_skills = list(dict.fromkeys(normalized_skills))  # This preserves order
    
    # Map the normalized skills back to the original names
    unique_top_skills = [unique_skills[normalized_skills.index(skill)] for skill in unique_normalized_skills]
    
    # Return the top N unique skills
    return unique_top_skills[:top_n]



def get_scenarios(job_description):
    jd_skills = extract_skills_from_jd(job_description)
    matching_skills = find_top_matching_skills(jd_skills)
    print('jd_skills:', jd_skills, flush=True)
    print('matching_skills:', matching_skills, flush=True)
    
    # Get scenarios matching those skills
    scenarios = []
    for skill in matching_skills:
        matched = scenarios_df[scenarios_df["Skill Name"] == skill]["Scenarios"].dropna().tolist()
        for scenario in matched:
            scenarios.append((skill, scenario))
    return scenarios


def scenario_chatbot_response(user_input, job_title, job_description, scenario_text):
    scenario_history.append({"role": "user", "content": user_input})
    
    prompt = f"""You are an AI interviewer conducting a scenario-based behavioral interview.
    
    Job Title: {job_title}
    Job Description: {job_description}
    Scenario (to be tailored to the job): {scenario_text}
    
    Ask one question at a time based on the candidate's responses. 
    Each follow-up should dig deeper into their reasoning or experience. 
    Use the job context above to make the scenario and follow-ups more relevant.
    Do not ask multiple questions at once. 
    Keep it professional and focused."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt},
                 {"role": "user", "content": user_input}],
        temperature=0.7,
        max_tokens=200
    )
    
    bot_reply = response.choices[0].message.content.strip()
    scenario_history.append({"role": "assistant", "content": bot_reply})
    
    return bot_reply



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

@app.route('/get_job_description', methods=['GET'])
def get_job_description():
    global current_job_description 
    current_job_description = generate_job_description(current_job_title)
    global current_scenario_text
    current_scenario_text = get_scenarios(current_job_description)
    return jsonify({
        'job_title': current_job_title,
        'job_description': current_job_description
    })

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

@app.route('/ask_scenario', methods=['POST'])
def ask_scenario():
    user_input = request.form['user_input']
    # scenario_text = get_scenarios(current_job_description)
    # print(scenario_text)

    # Process the user input and generate a scenario-based response
    bot_response = scenario_chatbot_response(user_input, current_job_title, current_job_description, current_scenario_text)
    audio_url = generate_speech(bot_response)  # Reuse existing TTS function

    return jsonify({
        'response': bot_response,
        'audio': audio_url,
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
    port = int(os.getenv("PORT", 8080))
    os.makedirs("static/audio", exist_ok=True)
    # socketio.run(app, host='127.0.0.1', port=5000)
    socketio.run(app, host='0.0.0.0', port=port, debug=True)

