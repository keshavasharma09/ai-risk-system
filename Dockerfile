# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements first (to leverage Docker layer caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . /app

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Start the Uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]