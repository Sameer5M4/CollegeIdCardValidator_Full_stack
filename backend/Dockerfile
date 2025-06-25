FROM python:3.9-slim

# ✅ CORRECTED: Install OS packages including curl for the healthcheck
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port & run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]