# Use NVIDIA NeMo official image - has pre-compiled extensions for latest GPUs
FROM nvcr.io/nvidia/nemo:25.09.01

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directories early (cached)
RUN mkdir -p /app/uploads /app/outputs /app/models /app/app/static

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Python files
COPY app/*.py ./app/

# Copy static files
COPY app/static/ ./app/static/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
