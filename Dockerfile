# Betfair Trading Bot Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install understat with --no-deps to avoid dependency conflicts
# (library has overly pinned deps but works fine with our versions)
RUN pip install --no-cache-dir --no-deps understat>=0.1.14

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

# Create data directories
RUN mkdir -p /app/data/logs && \
    chown -R botuser:botuser /app/data

# Switch to non-root user
USER botuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TRADING_MODE=paper

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "scripts/run_paper_trading.py"]
