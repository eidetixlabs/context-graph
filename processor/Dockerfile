FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY processor/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from pydantic_ai.mcp import MCPServerStreamableHTTP"

# Pre-fetch spaCy model so the container starts without a network call.
RUN python -c "import spacy; spacy.load('en_core_web_sm')"

COPY base/ ./base/
COPY processor/ ./processor/

ENTRYPOINT ["python", "-m"]
CMD ["processor.worker", "--provider", "trace"]
