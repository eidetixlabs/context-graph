# Security

## Reporting

If you find a vulnerability in Context Graph, email the maintainers privately.
Do not open a public issue for secrets, auth bypasses, or injection bugs.

## Local secrets

- Put `OPENAI_API_KEY` and `API_KEY` in `.env` (never commit it).
- `API_KEY` is a bearer secret. Anyone with it can read and write your local graph.
- Do not paste `~/.context-graph/` files into bug reports.

## Scope

This project is meant for local and trusted-network use. It does not ship
multi-tenant SaaS hardening, license checks, or hosted identity.
