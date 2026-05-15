# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ─── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─── Copy source code ─────────────────────────────────────────────────────────
COPY src/     ./src/
COPY config/  ./config/

# ─── Set PYTHONPATH ────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app

# ─── Entry point (overridden by KFP component at runtime) ─────────────────────
CMD ["python", "-m", "src.pipelines.full_pipeline", "--help"]