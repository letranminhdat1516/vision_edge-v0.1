# Vision Edge Healthcare System - Docker Image for Raspberry Pi
FROM python:3.10-slim-bullseye

# Install system dependencies (including audio libs for pygame)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libpq-dev \
    gcc \
    g++ \
    # Audio dependencies for pygame
    libsdl2-mixer-2.0-0 \
    libsdl2-2.0-0 \
    libasound2-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env .env
COPY yolov8n-pose.pt .
COPY yolov8s.pt .

# Create necessary directories
RUN mkdir -p data/saved_frames/alerts examples/data/saved_frames/alerts

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
# Disable Qt GUI (headless mode) - FIX xcb error
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=
ENV OPENCV_VIDEOIO_DEBUG=0
# Disable SDL video for pygame audio-only
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=alsa

# Run the application
CMD ["python", "src/main.py"]
