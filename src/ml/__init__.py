"""
Machine Learning Infrastructure for T-Bot Trading System

This module provides comprehensive ML infrastructure for model training, versioning,
inference, and lifecycle management for financial trading applications.

Key Components:
- Model Manager: Central ML lifecycle management
- Feature Engineering: Feature creation and selection
- Training: Model training orchestration
- Inference: Real-time and batch predictions
- Registry: Model versioning and storage
- Validation: Model validation and drift detection

Dependencies:
- P-001: Core types, exceptions, config
- P-002A: Error handling framework
- P-007A: Utility decorators
"""

from .feature_engineering import FeatureEngineeringService
from .model_manager import ModelManagerService

__version__ = "1.0.0"
__all__ = [
    "FeatureEngineeringService",
    "ModelManagerService",
]
