FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY webserver/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY base/ ./base/
COPY connectors/ ./connectors/
COPY webserver/main.py ./main.py
COPY webserver/entrypoint.sh ./entrypoint.sh

RUN chmod +x entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
