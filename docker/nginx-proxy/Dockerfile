# Use Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    git \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install Backend dependencies
RUN uv sync --frozen

# Install Frontend dependencies
WORKDIR /app/src/frontend
RUN npm install

# Copy Nginx config
COPY docker/nginx-proxy/nginx.conf /etc/nginx/nginx.conf

# Set executable permissions for start script
WORKDIR /app
RUN chmod +x docker/nginx-proxy/start.sh

# Expose port 80 (Nginx)
EXPOSE 80

# Environment variables
ENV LANGFLOW_HOST=0.0.0.0
ENV LANGFLOW_PORT=7860
ENV VITE_PORT=3000

# Run start script
CMD ["./docker/nginx-proxy/start.sh"]
