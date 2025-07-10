FROM python:3.10-slim

# Set environment variables to reduce output and avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependencies for dlib and necessary libraries for Django
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-system-dev \
    python3-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install dlib, face_recognition, and other necessary Python packages
RUN pip install dlib face_recognition

# Copy the application files to the container
COPY . .

# Install Django and remaining Python dependencies
RUN pip install -r requirements.txt

# Expose a port if required by your application (optional)
EXPOSE 8000

# Command to run your Django application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
