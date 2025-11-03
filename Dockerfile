# Use a lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY server ./server
COPY artifacts ./artifacts

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to start the API
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
