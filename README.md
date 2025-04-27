OZONETELL_MAIN

Overview
OZONETELL_MAIN is a Flask-based web application designed to process audio files stored in IBM Cloud Object Storage (COS), transcribe them using the Assisto API, analyze content with the Watsonx API (for speaker separation, call quality, and insights), and generate downloadable JSON outputs. The app is containerized for deployment on IBM Code Engine, ensuring scalability and security.

Features

Retrieve and validate JSON metadata from IBM COS.
Download and transcribe audio files using the Assisto API.
Separate speakers, assess call quality, and extract insights using Watsonx API.
Generate and serve sanitized JSON outputs for download.
Secure session management and CSRF protection.
Deployable on IBM Code Engine with a Docker container.
Prerequisites
Python 3.9 or higher.
Docker installed for containerization.
IBM Cloud account with COS and API keys for Watsonx and Assisto.
ibmcloud CLI for Code Engine deployment.

Project Structure
OZONETELL_MAIN/
├── temp/              # Temporary files (e.g., audio, JSON)
├── templates/         # HTML templates (e.g., index.html)
├── venv/              # Virtual environment (optional for local setup)
├── .dockerignore      # Files to exclude from Docker build
├── .env               # Environment variables (not committed)
├── .gitignore         # Excludes sensitive files from version control
├── app.log            # Application log file
├── Dockerfile         # Docker configuration for deployment
├── gunicorn_config.py # Gunicorn server configuration
├── requirements.txt   # Python dependencies


Setup Instructions
Local Development
Clone the Repository:
git clone <repository-url>
cd OZONETELL_MAIN
Create a Virtual Environment (optional):
python -m venv venv
source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)
Install Dependencies:
pip install -r requirements.txt
Configure Environment Variables:
Create a .env file in the root directory with the following keys:
FLASK_SECRET_KEY=your-secure-random-key
COS_API_KEY=your-cos-api-key
COS_SERVICE_INSTANCE_ID=your-cos-service-id
COS_ENDPOINT=your-cos-endpoint
COS_BUCKET=ozonetell
ASSISTO_API_URL=your-assisto-api-url
WATSONX_API_KEY=your-watsonx-api-key
WATSONX_URL=your-watsonx-url
WATSONX_PROJECT_ID=your-project-id
WATSONX_MODEL_ID=your-model-id
WATSONX_AUTH_URL=your-auth-url
Ensure .env is listed in .gitignore.
Run the Application:
python app.py
Access the app at http://localhost:8000
Deployment to IBM Code Engine
Build and Push Docker Image:
docker build -t icr.io/your-namespace/flask-app:latest .
docker push icr.io/your-namespace/flask-app:latest
Create ConfigMap for Environment Variables:
ibmcloud ce configmap create --name flask-config --from-env-file .env
Deploy the Application:
ibmcloud ce app create --name flask-app --image icr.io/your-namespace/flask-app:latest --port 8000 --env-from-configmap flask-config --min-scale 1 --max-scale 10
Note the public URL provided in the output.
Usage
Visit the deployed URL or http://localhost:8000.
Select a JSON file from the COS bucket via the web form.
Submit to process the audio, generating a transcript, speaker separation, call quality assessment, and insights.
Download the resulting JSON file using the provided link.
Functions Overview
get_cos_files(): Retrieves and validates JSON files from COS.
process_audio_with_assisto(): Transcribes audio using the Assisto API.
Separatespeakers(): Separates speakers in the transcript with Watsonx.
Getcallquality(): Assesses call quality using Watsonx.
Getinsights(): Extracts insights and sentiment with Watsonx.
create_json_output(): Generates a sanitized JSON output file.
index(): Renders the main web interface and handles form submissions.
api_process(): Processes JSON via API endpoint.
download_json(): Serves the JSON output for download.
Testing
Run python test_audio_processing.py to execute unit tests for audio processing, API calls, and JSON generation.
Add new tests to cover edge cases (e.g., invalid COS files, API failures).
Security Notes
All outputs are sanitized with bleach to prevent XSS.
CSRF protection is enabled with Flask-WTF.
Credentials are managed via environment variables and ConfigMaps.
Sessions use secure cookies (SESSION_COOKIE_SECURE=True, HTTPONLY=True, SAMESITE=Lax).
Suggestions for Improvement
Add Flask-Limiter to rate-limit /api/process (e.g., 10 requests per minute).
Implement Flask-Login or OAuth for authentication on /download_json.
Add log rotation with RotatingFileHandler and integrate with IBM Log Analysis.
Sanitize COS JSON inputs (e.g., AudioFile, Apikey) with bleach or a schema validator.
Regularly scan the Docker image with IBM Vulnerability Advisor.
