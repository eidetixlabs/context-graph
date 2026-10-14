# Contributing

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r webserver/requirements.txt -r requirements-dev.txt
.venv/bin/pytest -q
```

Processor tests that import spaCy or provider SDKs need `processor/requirements.txt`.

## Pull requests

- Add tests for MCP, ingest, and processor behavior you change.
- Do not add license servers, private registries, or cloud account identifiers.
- Keep the default local path `docker compose up` working without AWS credentials.

## Code of conduct

Be respectful. Harassment or personal attacks are not welcome.
