"""
processor/ — Document processing pipeline for Eidetix cloud.

Polls Postgres for documents in 'raw' or 'scored' state and runs them
through scoring (small model) and extraction (larger model with MCP tools).

Entry point:
    python -m processor.worker --provider notion
"""
