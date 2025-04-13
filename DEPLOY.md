# Deployment Instructions for AI Content Detector

This application can be deployed to various cloud platforms:

## Streamlit Cloud
1. Push this code to a GitHub repository
2. Go to https://streamlit.io/cloud
3. Connect your GitHub account
4. Select the repository and branch
5. Set the main file path to: streamlit_app.py
6. Deploy!

## Render.com
1. Push this code to a GitHub repository
2. Go to https://dashboard.render.com/
3. Create a new Web Service
4. Connect your GitHub repository
5. Render will automatically detect the render.yaml configuration
6. Click "Create Web Service"

## Google App Engine
1. Install Google Cloud SDK
2. Run: gcloud init
3. Run: gcloud app deploy app.yaml

## Docker
1. Build the Docker image:
   docker build -t ai-content-detector .
2. Run the container:
   docker run -p 8501:8501 ai-content-detector
3. Access the application at http://localhost:8501

## Heroku
1. Install Heroku CLI
2. Login: heroku login
3. Create app: heroku create ai-content-detector
4. Push to Heroku: git push heroku main
