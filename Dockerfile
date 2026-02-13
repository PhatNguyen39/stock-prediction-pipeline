FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved logs mlruns

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Expose port (default 7860 for HF Spaces; Fly.io overrides via PORT env var)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Run the API using PORT env var (defaults to 7860 for HF Spaces)
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
