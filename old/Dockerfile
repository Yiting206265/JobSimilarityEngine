# Use an ARM64-compatible Python image
FROM python:3.9-slim-buster

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies for building Python packages from source
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libomp-dev \
    libffi-dev \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip

# Install spaCy, blis, and faiss-cpu using precompiled binaries to avoid build issues
RUN pip install --no-cache-dir --only-binary :all: spacy
RUN pip install -r requirements.txt

# Expose the port the app runs on (FastAPI typically uses port 8000)
EXPOSE 8000

# Set the default command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

