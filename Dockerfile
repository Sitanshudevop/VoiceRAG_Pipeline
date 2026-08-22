FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Generate the initial FAISS & BM25 indices
RUN python ingest.py

# Expose port required by Hugging Face Spaces
EXPOSE 7860

# Run FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
