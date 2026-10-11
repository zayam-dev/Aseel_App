FROM python:3.10-slim

WORKDIR /app

# OpenCV ke zaroori Linux packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Requirements install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baqi sara project code copy
COPY . .

# Hugging Face Spaces port 7860 use karta hai
EXPOSE 7860

CMD ["python", "app.py"]