#!/bin/bash

# Exit on error
set -e

echo "Starting Backend (FastAPI)..."
uv run langflow run --backend-only --env-file .env --host 0.0.0.0 &

echo "Starting Frontend (Vite)..."
cd src/frontend
# Using dev:docker script which has --host 0.0.0.0
npm run start -- --host 0.0.0.0 &

echo "Starting Nginx Proxy..."
# Nginx needs to stay in the foreground to keep the container running
nginx -g 'daemon off;'
