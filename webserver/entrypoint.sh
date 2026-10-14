#!/usr/bin/env bash
set -euo pipefail

_read_secret() {
    local secret_file="/run/secrets/$1"
    if [[ -f "$secret_file" ]]; then
        tr -d '\n' < "$secret_file"
    elif [[ -n "${2:-}" && -n "${!2:-}" ]]; then
        echo -n "${!2}"
    fi
}

if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
    val=$(_read_secret neo4j_password)
    [[ -n "$val" ]] && export NEO4J_PASSWORD="$val"
fi

if [[ -z "${API_KEY:-}" ]]; then
    val=$(_read_secret api_key)
    [[ -n "$val" ]] && export API_KEY="$val"
fi

if [[ -z "${DATABASE_URL:-}" ]] && [[ -f "/run/secrets/postgres_password" ]]; then
    PG_PASS=$(tr -d '\n' < /run/secrets/postgres_password)
    PG_HOST="${POSTGRES_HOST:-postgres}"
    PG_PORT="${POSTGRES_PORT:-5432}"
    PG_USER="${POSTGRES_USER:-eidetix}"
    PG_DB="${POSTGRES_DB:-eidetix}"
    export DATABASE_URL="postgresql://${PG_USER}:${PG_PASS}@${PG_HOST}:${PG_PORT}/${PG_DB}"
fi

UVICORN_LOG_LEVEL="$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
exec uvicorn main:app --host 0.0.0.0 --port 8080 --log-level "$UVICORN_LOG_LEVEL" --no-access-log
