#!/usr/bin/env python3
"""
TA-Lib Installation Test Script
Tests TA-Lib installation and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mark this file as not a pytest test file
__test__ = False

import structlog

def setup_logging():
    """Setup structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def test_talib_installation():
    """Test TA-Lib installation and basic functionality."""
    # This is a standalone script, not a pytest test
    logger = structlog.get_logger(__name__)
    
    logger.info("🧠 TA-Lib Installation Test")
    logger.info("=" * 60)
    
    # Test 1: Check if TA-Lib is installed
    logger.info("📦 Checking TA-Lib installation...")
    try:
        import talib
        logger.info("✅ TA-Lib imported successfully")
        
        # Get version if available
        try:
            version = talib.__version__
            logger.info(f"📋 TA-Lib version: {version}")
        except AttributeError:
            logger.info("📋 TA-Lib version: unknown (version attribute not available)")
            
    except ImportError as e:
        logger.error("❌ TA-Lib not installed", error=str(e))
        return False
    
    # Test 2: Test basic TA-Lib functionality
    logger.info("🎯 Testing TA-Lib functionality...")
    try:
        import numpy as np
        
        # Create sample data
        close_prices = np.array([10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.8, 12.0], dtype=np.float64)
        high_prices = np.array([10.2, 10.7, 11.1, 10.9, 11.3, 11.6, 11.9, 12.1], dtype=np.float64)
        low_prices = np.array([9.8, 10.3, 10.9, 10.7, 11.1, 11.4, 11.7, 11.9], dtype=np.float64)
        volume = np.array([1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700], dtype=np.float64)
        
        logger.info("📊 Sample data created successfully")
        
        # Test SMA (Simple Moving Average)
        sma = talib.SMA(close_prices, timeperiod=3)
        logger.info("✅ SMA calculation successful", sma_result=sma.tolist())
        
        # Test RSI (Relative Strength Index)
        rsi = talib.RSI(close_prices, timeperiod=3)
        logger.info("✅ RSI calculation successful", rsi_result=rsi.tolist())
        
        # Test MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        logger.info("✅ MACD calculation successful", 
                   macd_result=macd.tolist(),
                   signal_result=macd_signal.tolist(),
                   histogram_result=macd_hist.tolist())
        
        # Test Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices)
        logger.info("✅ Bollinger Bands calculation successful",
                   upper_result=upper.tolist(),
                   middle_result=middle.tolist(),
                   lower_result=lower.tolist())
        
        # Test Stochastic
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
        logger.info("✅ Stochastic calculation successful",
                   slowk_result=slowk.tolist(),
                   slowd_result=slowd.tolist())
        
        logger.info("🎉 All TA-Lib functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error("❌ TA-Lib functionality test failed", error=str(e))
        return False

def test_talib_performance():
    """Test TA-Lib performance with larger datasets."""
    # This is a standalone script, not a pytest test
    logger = structlog.get_logger(__name__)
    
    logger.info("⚡ Testing TA-Lib performance...")
    
    try:
        import numpy as np
        import time
        
        # Create larger dataset
        data_size = 10000
        close_prices = np.random.random(data_size) * 100 + 50  # Random prices between 50-150
        high_prices = close_prices + np.random.random(data_size) * 5
        low_prices = close_prices - np.random.random(data_size) * 5
        
        logger.info(f"📊 Created dataset with {data_size} data points")
        
        # Performance test: SMA calculation
        start_time = time.time()
        sma = talib.SMA(close_prices, timeperiod=20)
        sma_time = time.time() - start_time
        
        # Performance test: RSI calculation
        start_time = time.time()
        rsi = talib.RSI(close_prices, timeperiod=14)
        rsi_time = time.time() - start_time
        
        # Performance test: MACD calculation
        start_time = time.time()
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        macd_time = time.time() - start_time
        
        logger.info("✅ Performance test completed",
                   sma_time=f"{sma_time:.4f}s",
                   rsi_time=f"{rsi_time:.4f}s", 
                   macd_time=f"{macd_time:.4f}s",
                   total_time=f"{sma_time + rsi_time + macd_time:.4f}s")
        
        return True
        
    except Exception as e:
        logger.error("❌ TA-Lib performance test failed", error=str(e))
        return False

def main():
    """Main test function."""
    setup_logging()
    logger = structlog.get_logger(__name__)
    
    logger.info("🧠 TA-Lib Installation Verification")
    logger.info("=" * 60)
    
    # Run tests
    installation_success = test_talib_installation()
    performance_success = test_talib_performance()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 SUMMARY")
    logger.info("=" * 60)
    
    if installation_success and performance_success:
        logger.info("✅ TA-Lib installation test passed")
        logger.info("💡 TA-Lib is properly installed and functional")
    else:
        logger.error("❌ TA-Lib installation test failed")
        logger.info("💡 Next steps:")
        logger.info("   - Run: ./scripts/setup/talib.sh install")
        logger.info("   - Check system dependencies")
        logger.info("   - Verify C++ compiler availability")
    
    return installation_success and performance_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 