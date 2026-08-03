# ---------------------------------------------------------------------------
# Fitness-assistant – production container image
#
# Build:  docker build -t fitness-assistant .
# Run:    docker compose up
# ---------------------------------------------------------------------------

# Use the slim variant to minimise image size.
FROM python:3.12-slim

# Install uv (the fast Python package manager used by this project).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Dependency installation ─────────────────────────────────────────────────
# Copy only the files needed to resolve + install dependencies first so that
# Docker can cache this layer and skip re-installation on code-only changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# ── Application source ──────────────────────────────────────────────────────
COPY src/ ./src/
COPY data/ ./data/

# Make the fitness_assistant package importable from /app/src.
ENV PYTHONPATH="/app/src/fitness_assistant"
ENV PATH="/app/.venv/bin:$PATH"

# Do not write .pyc files inside the container (keeps layers clean).
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure stdout/stderr are not buffered (logs appear in real time).
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "-m", "api.app"]
