# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (useful for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# 5. Install Python dependencies without caching to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code, including models and processed data
COPY . .

# 7. Expose the port that FastAPI will use
EXPOSE 8000

# 8. Command to start the uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]