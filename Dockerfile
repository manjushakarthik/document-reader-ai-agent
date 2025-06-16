FROM python:3.10-slim

# Set environment variables
ENV PORT=8080
ENV STREAMLIT_WATCHER_TYPE=none

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app_clean.py .

# Expose port
EXPOSE 8080

# Start Streamlit app
CMD ["streamlit", "run", "app_clean.py", "--server.address=0.0.0.0", "--server.port=8080", "--server.enableCORS=false"]
