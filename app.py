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
from google.cloud import texttospeech


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

scenario_index = 0
follow_up_count = 0
scenario_scores = []
follow_up_question = ""

current_job_title = "AI Engineer"
conversation_history = []
scenario_history = []


def generate_job_description(job_title):
    prompt = f"""
    Write a detailed job description for a {job_title}. Include responsibilities, qualifications, and key skills required.

    Do not use markdown or formatting symbols (such as **, ##, -, etc.). Just write clean plain text in full sentences and paragraphs.
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def extract_skills_from_jd(jd_text):
    prompt = f"""Extract the top 5 most relevant soft/behavioral skills from this job description:
    
    {jd_text}
    
    List them one per line concise phrases (2–3 words max). 
    For example: Communication, Teamwork, Problem Solving, Adaptability, Leadership."""
    
    response = client.chat.completions.create(
        model="gpt-4.1",
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


def generate_scenario_text(job_title, job_description, scenario_text):
    prompt = f"""
        You are an expert behavioral interviewer.

        Your task:
        - Reframe the following scenario so that it feels realistic and relevant to someone interviewing for a **{job_title}** role.
        - Use the job description below to guide your rewrite.
        - Keep the tone conversational and professional.
        - Do NOT include any questions — just describe the adapted situation in 3–4 sentences max.
        - Avoid repeating the original scenario text word-for-word.
        - Make it feel specific to this job context.

        Job Description:
        {job_description}

        Original Scenario:
        "{scenario_text}"

        Now rewrite the scenario accordingly:
        """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=180
    )

    return response.choices[0].message.content.strip()



def scenario_chatbot_response(user_input, job_title, job_description, scenario_text, follow_up_count):
    scenario_history.append({"role": "user", "content": user_input})

    if follow_up_count == 0:
        prompt = f"""You’re a professional interviewer speaking to a candidate for a **{job_title}** role.

        Job Description:
        {job_description}

        Here’s the situation:
        {scenario_text}

        Your task:
        - Introduce this situation naturally to the candidate.
        - Keep your tone conversational, warm, and clear — like you're genuinely curious about their thinking.
        """

    else:
        prompt = f"""You’re continuing a behavioral interview for a **{job_title}** position.

        Job Description:
        {job_description}

        Original situation (for your reference):
        {scenario_text}

        Candidate just said:
        "{user_input}"

        Your task:
        - Ask **one** follow-up question to understand more about their decisions, thought process, or the outcome.
        - Avoid repeating the scenario or previous answers.
        - Keep your tone professional and curious — like a thoughtful human interviewer.
        """

    messages = [
        {"role": "system", "content": prompt},
        *scenario_history[-4:],
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.6,
        max_tokens=200
    )

    bot_reply = response.choices[0].message.content.strip()
    scenario_history.append({"role": "assistant", "content": bot_reply})

    return bot_reply




def score_answer(scenario_text, follow_up_question, user_answer):
    scoring_prompt = f"""
    You are an AI interviewer. Evaluate the candidate's response to a scenario-based question.

    Scenario: {scenario_text}
    Interviewer Follow-Up: {follow_up_question}
    Candidate Response: {user_answer}

    Score the response from 0 to 2:
    - 0 = Inadequate or irrelevant
    - 1 = Somewhat relevant or incomplete
    - 2 = Strong and relevant

    Respond with a single integer (0, 1, or 2).
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": scoring_prompt}],
        temperature=0,
        max_tokens=5
    )

    score_text = response.choices[0].message.content.strip()
    try:
        return int(re.search(r'\d', score_text).group())
    except:
        return 0


def chatbot_response_with_history(user_input, job_description=None):
    conversation_history.append({"role": "user", "content": user_input})

    if job_description is None:
        job_description = "the role the user is applying for."

    # Generate the interview-specific system prompt
    prompt = f"""
    You are an experienced AI interviewer conducting a structured behavioral interview for the role of **{current_job_title}**.

    Use this job description to guide your questions:
    {job_description}

    Your task:
    - Ask **one** well-phrased, relevant question at a time.
    - Base your questions on the candidate's resume, past responses, and the job description.
    - Keep the tone professional and focused on uncovering the candidate’s qualifications and thought process.
    - Avoid casual chatter or vague open-ended prompts.
    - Do **not** ask multiple questions at once or rephrase the same question.
    """

    
    response = client.chat.completions.create(
        model="gpt-4.1", 
        messages=[
            {"role": "system", "content": prompt},
            *conversation_history[-4], #only last few exchanges to stay focused
            {"role": "user", "content": user_input},
        ],
        max_tokens=200,  # Adjust for longer/shorter responses
        temperature=0.5,  # Controls the creativity of the response
    )
    bot_reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": bot_reply})  # Save bot response

    return bot_reply

    
def generate_speech(text):
    """Convert text response to speech using OpenAI TTS."""
    instructions = """Voice Affect: Professional, confident, and attentive. Demonstrates authority without being intimidating.

Tone: Supportive, thoughtful, and respectful. Curious about the candidate’s experiences and insights.

Pacing: Steady and deliberate. Slightly slower when asking complex or reflective questions, allowing candidates time to process.

Emotions: Calm interest, encouragement, and professionalism.

Pronunciation: Clear and articulate. Emphasize key phrases like “decision-making,” “impact,” or “teamwork.” """
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="verse", 
        instructions = instructions,
        input=text,
        speed=1.15
    )
    
    audio_path = "static/audio/response.mp3"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(response.content)

    return audio_path


@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return {'error': 'No audio file provided'}, 400

    audio_file = request.files['audio']
    
    # Save to temp file
    temp_path = "temp_audio.webm"
    audio_file.save(temp_path)

    # Transcribe using OpenAI 
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",  # or "gpt-4o"
        file=open(temp_path, "rb"),
        response_format="text",
        language="en"
    )

    return {'transcript': transcription}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_job_description', methods=['GET'])
def get_job_description():
    global current_job_description 
    current_job_description = generate_job_description(current_job_title)
    global scenario_list
    scenario_list = get_scenarios(current_job_description)
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
    global scenario_index, follow_up_count, follow_up_question
    
    user_input = request.form['user_input']
    
    # If done with all scenarios
    if scenario_index >= len(scenario_list):
        average_score = sum(scenario_scores) / len(scenario_scores)
        
        return jsonify({
            'response': "Thank you, that concludes the interview.",
            'audio': generate_speech("Thank you, that concludes the interview."),
            'average_score': average_score
        })

    current_scenario_text = scenario_list[scenario_index][1]
    
    revised_scenario_text = generate_scenario_text(
        current_job_title, 
        current_job_description, 
        current_scenario_text
    )
    
    score = 0
    # Score the previous answer using the stored follow-up question
    if follow_up_question:
        score = score_answer(current_scenario_text, follow_up_question, user_input)
        scenario_scores.append(score)


    bot_response = scenario_chatbot_response(
        user_input,
        current_job_title,
        current_job_description,
        revised_scenario_text,
        follow_up_count
    )
    
    # Save the follow-up question for scoring next time
    follow_up_question = bot_response

    # Update follow-up count and scenario index
    follow_up_count += 1
    if follow_up_count >= 3:
        follow_up_count = 0
        scenario_index += 1

    audio_url = generate_speech(bot_response)  # Reuse existing TTS function

    return jsonify({
        'response': bot_response,
        'audio': audio_url,
        'score': score })


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

