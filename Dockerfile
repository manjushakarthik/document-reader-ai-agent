# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8080

# Run Streamlit app, using the port provided by the environment (default 8080)
CMD streamlit run any_document.py --server.address=0.0.0.0 --server.port=${PORT} --server.enableCORS=false
