FROM python:3.10-slim

# Everything happens inside /app
WORKDIR /app

# System dependencies (needed for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

EXPOSE 8000

# Production-safe start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
