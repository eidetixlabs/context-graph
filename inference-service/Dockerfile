FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY inference-service/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY inference-service/main.py ./main.py

EXPOSE 9111
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9111"]
