FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Create non-root user required by HF Spaces
RUN useradd -m -u 1000 user

# Install CPU-only torch first (saves ~1.5 GB vs the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (pip skips torch since it's already satisfied)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and give ownership to the non-root user
COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    USE_HF_STORAGE=true \
    HF_DATASET_REPO=Coupur/distrack-data

EXPOSE 7860 8000

CMD ["bash", "/app/startup.sh"]
