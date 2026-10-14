"""
processor/extraction/ — Hybrid NER extraction subpackage.

Provides fast, layered entity extraction:
  - Level 1: spaCy NER (PERSON, ORG, GPE)
  - Level 2: GLiNER zero-shot NER (service, tool, event, repository, …)
  - LLM enrichment layer: structured-output enrichment of raw candidates
"""
