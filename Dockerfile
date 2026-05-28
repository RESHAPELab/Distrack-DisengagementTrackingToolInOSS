FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    USE_HF_STORAGE=true \
    HF_DATASET_REPO=SamUtz1/distrack-data

EXPOSE 7860

CMD ["bash", "/app/startup.sh"]
