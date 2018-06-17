FROM python:2.7.15-slim-stretch
WORKDIR /app
COPY main.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "/app/main.py"]
