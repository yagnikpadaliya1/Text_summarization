FROM python:3.10-slim

WORKDIR /app

# Install sentencepiece explicitly first (required for T5 tokenizer)
RUN pip install --no-cache-dir sentencepiece

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
