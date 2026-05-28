#!/bin/bash
set -e

# Run the DISTRAC FastAPI app on port 7860 (the port HF Spaces exposes publicly)
cd /app/Dashboard
exec uvicorn distrac_api:app --host 0.0.0.0 --port 7860
