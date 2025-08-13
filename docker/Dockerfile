# Trading Bot Dockerfile with NVIDIA CUDA support for RTX 4090
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# GPU support environment variables for RTX 4090
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install TA-Lib from source
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Copy requirements files
COPY requirements/ requirements/

# Install Python dependencies with rapids packages for RTX 4090
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r requirements/development.txt \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Run GPU detection and log results
RUN python3 -c "from src.utils.gpu_detector import detect_and_log_gpu_environment; detect_and_log_gpu_environment()"

# Create necessary directories
RUN mkdir -p logs/application \
    logs/system \
    logs/exchange \
    logs/database \
    logs/ml \
    data/raw \
    data/processed \
    data/features \
    data/cache \
    models/trained_models \
    models/model_registry \
    state/sessions \
    state/recovery \
    state/temp \
    backups/database \
    backups/config \
    reports/performance \
    reports/strategy \
    reports/risk

# Set proper permissions
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "src.web_interface.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 