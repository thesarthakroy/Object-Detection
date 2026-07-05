# Base image
FROM python:3.9-slim

# Prevent python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install python requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy directories and configuration files
COPY models/ ./models/
COPY core/ ./core/
COPY utils/ ./utils/
COPY components/ ./components/
COPY app.py .

# Create output folder for logging/CSV export
RUN mkdir -p output

# Expose Streamlit default port
EXPOSE 8501

# Run the streamlit application by default
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
