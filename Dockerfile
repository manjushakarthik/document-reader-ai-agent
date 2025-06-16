FROM python:3.10-slim

ENV PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_clean.py .

EXPOSE 8080

CMD streamlit run app_clean.py --server.address=0.0.0.0 --server.port=${PORT} --server.enableCORS=false