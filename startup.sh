#!/bin/bash
set -e

# Start FastAPI backend on port 8000 (background)
cd /app/Dashboard
uvicorn distrac_api:app --host 0.0.0.0 --port 8000 &

# Give the API a moment to bind before Streamlit starts
sleep 3

# Start Streamlit frontend on port 7860 (foreground — keeps the container alive)
# HF Spaces exposes port 7860 to the public URL
streamlit run /app/Dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true
