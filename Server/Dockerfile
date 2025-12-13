FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from Server/
COPY Server/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire Server directory
COPY Server .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
