FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p .streamlit

# Copy Streamlit config
COPY .streamlit/config.toml .streamlit/

# Expose port for the application
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
