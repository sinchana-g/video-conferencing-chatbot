# Video Conferencing Chatbot

This is a video conferencing chatbot application that uses OpenAI's GPT-4 for conducting job interviews. It includes a scenario-based interview, job description generation, and real-time emotion detection. The application uses Flask for the backend, Flask-SocketIO for WebRTC signaling, and OpenAI for conversational responses and TTS (Text-to-Speech).

## Prerequisites

Before running the application, make sure you have the following installed on your machine:

- **Python 3.7** or above
- **Docker** (if you want to run the app in a container)
- **Flask** and other Python dependencies

## Setup Instructions

### 1. Install Dependencies

First, create a virtual environment and install the required dependencies.

    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

    # Install the dependencies from requirements.txt
    pip install -r requirements.txt
    

### 2. Set Up OpenAI API Key

The application requires an OpenAI API key for generating job descriptions and interview questions.

1. Create a `.env` file in the root directory of the project.
2. Add the following line to your `.env` file:

   ```bash
   OPENAI_API_KEY=your-openai-api-key-here

Make sure to replace your-openai-api-key-here with your actual OpenAI API key.

### 3. Running the Application Locally

To run the Flask app locally, use the following command:

    ```bash
    # Run the Flask app
    python app.py

   
The application will be available at http://127.0.0.1:5000/.


### 4. Docker Setup (Optional)

If you prefer to run the application inside a Docker container, you can create a Docker image, build the Docker image and Run the Docker container. 

    ```bash
    docker build -t video-conferencing-chatbot .
    docker run -p 8080:8080 video-conferencing-chatbot
    
    
The application will be available at http://localhost:8080/.


## How it Works

1. **Job Description Generation**: 
   - The application uses OpenAI's GPT-4 to generate a detailed job description based on a job title. The job description includes key responsibilities, qualifications, and skills required.
   
2. **Skill Extraction**: 
   - From the generated job description, the application extracts the top 5 soft skills and behavioral traits using GPT-4.
   
3. **Skill Matching**: 
   - The extracted skills are matched with predefined scenarios that are relevant to each skill. This matching process ensures that the questions asked during the interview are directly related to the job description and the required skills.

4. **Interview Modes**:
   - **Normal Interview**: The chatbot conducts a standard job interview based on the generated job description, asking one relevant question at a time and following up based on the candidate’s responses.
   - **Scenario-based Interview**: The chatbot asks behavioral interview questions based on job-related scenarios, testing the candidate on specific skills. This mode also includes follow-up questions to dive deeper into the candidate’s responses.

5. **Voice Feedback**:
   - Responses from the chatbot are converted into speech using OpenAI's Text-to-Speech (TTS) model. This allows for a more natural interaction between the candidate and the chatbot.

6. **WebRTC for Real-Time Communication**:
   - The application uses WebRTC (via Flask-SocketIO) for real-time video communication between the user and the system. This enables live interview-like sessions where the user can interact with the AI in real time.
