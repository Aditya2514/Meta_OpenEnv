# ── Assignment & Bug-Fix Planner Agent – Docker Image ──────────────────────
# Target: Hugging Face Spaces (Docker backend, tag `openenv`)
# Base:   python:3.11-slim  (small, reproducible, no unnecessary extras)
# Port:   7860 (HF Spaces default)
#
# Local build & test:
#   docker build -t assignment-planner -f src/envs/assignment_planner/server/Dockerfile .
#   docker run -p 7860:7860 assignment-planner
#
# Then visit: http://localhost:7860/docs
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
# No native extensions needed; keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to exploit Docker layer cache.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        python-dotenv \
        pyyaml \
    && pip install --no-cache-dir -r /app/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# Copy the assignment_planner package into /app/assignment_planner so that
# `from assignment_planner import ...` works without installing the package.
COPY src/envs/assignment_planner/__init__.py        /app/assignment_planner/__init__.py
COPY src/envs/assignment_planner/models.py          /app/assignment_planner/models.py
COPY src/envs/assignment_planner/task_config.py     /app/assignment_planner/task_config.py
COPY src/envs/assignment_planner/environment.py     /app/assignment_planner/environment.py
COPY src/envs/assignment_planner/graders.py         /app/assignment_planner/graders.py

# Copy the server entry-point and OpenEnv spec
COPY src/envs/assignment_planner/server/app.py      /app/app.py
COPY src/envs/assignment_planner/server/openenv.yaml /app/openenv.yaml

# ── Environment variables ─────────────────────────────────────────────────────
# PORT and DEFAULT_TASK can be overridden at `docker run` time:
#   docker run -e DEFAULT_TASK=hard_1 -p 7860:7860 assignment-planner
ENV PORT=7860
ENV DEFAULT_TASK=easy_1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Non-root user (HF Spaces security best practice) ─────────────────────────
RUN useradd -m -u 1000 appuser
USER appuser

# ── Health-check ───────────────────────────────────────────────────────────────
# HF Spaces polls / to verify the container is up.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
