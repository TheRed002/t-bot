# Technical Dependencies and Versions

## Overview
This document specifies exact dependency versions to ensure consistency across all prompts. These versions are referenced throughout the implementation.

## Core Framework Dependencies

### Python & Core Libraries
```
python>=3.11.0
pydantic==2.5.0
structlog==23.2.0
python-dotenv==1.0.0
asyncio-compat==0.2.0
```

### Web Framework & API
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
httpx==0.25.2
python-multipart==0.0.6
```

### Database & Storage
```
# PostgreSQL
sqlalchemy[asyncio]==2.0.23
alembic==1.13.0
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Redis
redis[hiredis]==5.0.1
aioredis==2.0.1

# InfluxDB
influxdb-client[async]==1.38.0
```

## Exchange Integration Dependencies

### Exchange APIs
```
ccxt==4.1.64
python-binance==1.0.19
websocket-client==1.6.4
```

### Networking & HTTP
```
aiohttp==3.9.1
requests==2.31.0
urllib3==2.1.0
certifi==2023.11.17
```

## Machine Learning Dependencies

### Core ML Libraries
```
# Scientific Computing
numpy==1.25.2
pandas==2.1.4
scipy==1.11.4

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0

# Deep Learning
tensorflow==2.15.0
torch==2.1.1
transformers==4.36.0

# Model Management
mlflow==2.8.1
optuna==3.4.0
```

### Technical Analysis
```
TA-Lib==0.4.28
pandas-ta==0.3.14b0
```

## Data Processing Dependencies

### Data Handling
```
# Financial Data
yfinance==0.2.28
alpha-vantage==2.3.1

# Alternative Data
tweepy==4.14.0
praw==7.7.1
newsapi-python==0.2.7

# Data Processing
arrow==1.3.0
pytz==2023.3.post1
```

## Testing Dependencies

### Testing Framework
```
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
pytest-benchmark==4.0.0
factory-boy==3.3.0
```

### Performance Testing
```
locust==2.17.0
memory-profiler==0.61.0
```

## Code Quality Dependencies

### Linting & Formatting
```
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0
bandit==1.7.5
```

### Type Checking
```
types-redis==4.6.0.11
types-requests==2.31.0.10
types-python-dateutil==2.8.19.14
```

## Monitoring & Observability

### Metrics & Monitoring
```
prometheus-client==0.19.0
grafana-api==1.0.3
```

### Logging & Tracing
```
sentry-sdk[fastapi]==1.38.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
```

## Deployment Dependencies

### Containerization
```
docker==6.1.3
docker-compose==1.29.2
```

### Process Management
```
supervisor==4.2.5
gunicorn==21.2.0
```

## Development Dependencies

### Development Tools
```
jupyter==1.0.0
ipython==8.17.2
jupyterlab==4.0.9
notebook==7.0.6
```

### Debugging
```
pdb++==0.10.3
ipdb==0.13.13
```

## Security Dependencies

### Cryptography & Security
```
cryptography==41.0.8
bcrypt==4.1.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pyotp==2.9.0
```

### Authentication
```
python-multipart==0.0.6
python-jose==3.3.0
```

## Environment-Specific Versions

### Development Environment
```python
# requirements/development.txt
-r base.txt

# Additional development tools
ipython==8.17.2
jupyter==1.0.0
pytest-watch==4.2.0
```

### Production Environment  
```python
# requirements/production.txt
-r base.txt

# Production server
gunicorn==21.2.0
supervisor==4.2.5

# Monitoring
sentry-sdk[fastapi]==1.38.0
prometheus-client==0.19.0
```

### Docker Environment
```python
# requirements/docker.txt  
-r production.txt

# Container-specific
docker==6.1.3
```

## Version Constraints & Compatibility

### Python Version Requirements
- **Minimum**: Python 3.11.0
- **Recommended**: Python 3.11.5+
- **Maximum**: Python 3.12.x (tested)

### Critical Version Pins
```
# These versions MUST match exactly
sqlalchemy==2.0.23  # Database compatibility
fastapi==0.104.1     # API compatibility  
tensorflow==2.15.0   # ML model compatibility
ccxt==4.1.64         # Exchange API compatibility
```

### Version Ranges (Acceptable)
```
# These can be updated within constraints
numpy>=1.25.0,<1.26.0
pandas>=2.1.0,<2.2.0
redis>=5.0.0,<6.0.0
```

## Installation Scripts

### Base Installation
```bash
# install_base_dependencies.sh
pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements/base.txt
```

### Development Setup
```bash
# install_dev_dependencies.sh  
pip install -r requirements/development.txt
pre-commit install
```

### Production Setup
```bash
# install_prod_dependencies.sh
pip install --no-dev -r requirements/production.txt
```

## Dependency Management Rules

### For All Prompts:
1. **Always use pinned versions** from this document
2. **Never install additional packages** without updating this document
3. **Import only specified versions** in requirements
4. **Test compatibility** before version changes

### Version Update Process:
1. Update this document first
2. Test with new versions
3. Update all requirements files
4. Validate across all components
5. Update prompts if needed

## Common Import Patterns

### Standard Imports by Category
```python
# Core Framework (P-001)
from pydantic import BaseModel, Field, BaseSettings
from src.core.logging import get_logger
from typing import Dict, List, Optional, Any

# Database (P-002) 
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

# Exchange Integration (P-003+)
import ccxt.pro as ccxt
from binance import AsyncClient
import websockets

# Machine Learning (P-017+)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import mlflow

# Web Framework (P-026+)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
import uvicorn

# Testing (P-033)
import pytest
from pytest_asyncio import fixture
from unittest.mock import AsyncMock, MagicMock
```

This document ensures all prompts use consistent, compatible dependency versions. 