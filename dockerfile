FROM python:3-trixie

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
