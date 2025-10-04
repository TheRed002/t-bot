"""
Test configuration utilities for production readiness tests.
"""

from typing import Any, Dict


class TestConfig:
    """Simple test configuration class that accepts dictionaries."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary."""
        self._config = config_dict
        
        # Set attributes from config dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, TestConfig(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config
    
    def __str__(self) -> str:
        """String representation without exposing secrets."""
        safe_config = self._config.copy()
        # Mask sensitive fields
        for key in ['api_secret', 'api_key', 'passphrase', 'password']:
            if key in safe_config:
                safe_config[key] = "***MASKED***"
        return str(safe_config)
    
    def __repr__(self) -> str:
        """Repr without exposing secrets."""
        return f"TestConfig({len(self._config)} keys)"