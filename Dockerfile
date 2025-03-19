FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
COPY app.py .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8570
CMD ["streamlit", "run", "app.py", "--server.port=8070", "--server.address=12.0.0.1server.rave=true"]
