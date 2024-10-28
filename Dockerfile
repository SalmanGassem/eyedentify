# Base image for running the app 
FROM python:3.10.14

RUN mkdir /app

WORKDIR /app

# Copy application code and resources
COPY app .
COPY app/pages .
COPY app/savedmodel .
COPY . .

# Install dependencies within a virtual environment
RUN pip install -r requirements.txt --no-cache-dir

# Expose Streamlit port (default 8501)
EXPOSE 8501

# Entrypoint to run the Streamlit app
CMD ["streamlit", "run", "./Home.py"]
