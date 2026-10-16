# syntax=docker/dockerfile:1
#
# Two deployable images from one file (see docs/deploy.md):
#
#   docker build --target app .   # the FastAPI service (esma-milan-server)
#   docker build --target web .   # Caddy: TLS, password, static frontend, /api proxy
#
# docker-compose.yml builds both. Nothing here names a host, user or
# password; those come from the .env file on the server at run time.

# --------------------------------------------------------------------------
# frontend-build: compile the React frontend to static files
# --------------------------------------------------------------------------
FROM node:22-alpine AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --------------------------------------------------------------------------
# app: the Python service
# --------------------------------------------------------------------------
FROM python:3.12-slim AS app

# uv, pinned to the version the repository is developed with.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# The repo root is the working directory: handlers.py and cli.py resolve
# the default taxonomy relative to it.
WORKDIR /app

# Dependencies first, so a change under src/ does not reinstall them.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# The application, plus the one file from the R reference it needs at
# run time, at the same relative path as in the repository.
COPY src/ ./src/
COPY "r_reference/inputs/ESMA template taxonomy.xlsx" "./r_reference/inputs/ESMA template taxonomy.xlsx"
RUN uv sync --frozen --no-dev

# Run as an unprivileged user.
RUN useradd --system --create-home --uid 10001 esma \
    && chown -R esma:esma /app
USER esma

EXPOSE 8000

# python:*-slim ships no curl; the standard library is enough for a probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

# The console script from the locked virtual environment, started directly:
# nothing is resolved or synced at run time, and uv is not needed after the
# build. Host, port and worker count come from ESMA_MILAN_* variables
# (docker-compose.yml and .env).
CMD ["/app/.venv/bin/esma-milan-server"]

# --------------------------------------------------------------------------
# web: Caddy with the built frontend
# --------------------------------------------------------------------------
FROM caddy:2-alpine AS web
COPY --from=frontend-build /build/dist /srv
COPY deploy/Caddyfile /etc/caddy/Caddyfile
