#!/usr/bin/env python3
"""
LightGBM CUDA Installation Test
Tests if LightGBM is properly installed with CUDA support.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
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

def test_lightgbm_installation():
    """Test LightGBM installation and GPU support."""
    # This is a standalone script, not a pytest test
    logger = structlog.get_logger(__name__)
    
    print("=" * 60)
    print("🔍 LightGBM CUDA Installation Test")
    print("=" * 60)
    
    # Test 1: Check if LightGBM is installed
    print("\n📦 Checking LightGBM installation...")
    
    # Activate virtual environment if available
    venv_path = os.path.expanduser("~/.venv")
    if os.path.exists(venv_path):
        # Add virtual environment to Python path
        venv_site_packages = os.path.join(venv_path, "lib", "python3.10", "site-packages")
        if os.path.exists(venv_site_packages):
            sys.path.insert(0, venv_site_packages)
            print(f"✅ Using virtual environment: {venv_path}")
    
    try:
        import lightgbm as lgb
        # Try to get version, but handle case where __version__ is not available
        try:
            version = lgb.__version__
        except AttributeError:
            version = "unknown (version attribute not available)"
        print(f"✅ LightGBM installed: {version}")
        logger.info("LightGBM installation verified", version=version)
    except ImportError as e:
        print(f"❌ LightGBM not installed: {e}")
        logger.error("LightGBM not installed", error=str(e))
        return False
    
    # Test 2: Check GPU support
    print("\n🎮 Testing GPU support...")
    
    # Create test data
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    data = lgb.Dataset(X, label=y)
    
    # Method 1: Try CUDA support (for NVIDIA GPUs)
    print("   Attempting CUDA training...")
    try:
        params_cuda = {
            'device': 'cuda',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_boost_round': 1,
            'verbose': -1
        }
        
        bst = lgb.train(params_cuda, data)
        print("✅ LightGBM CUDA training successful")
        logger.info("LightGBM CUDA support verified", device="cuda")
        
        # Test prediction
        predictions = bst.predict(X[:5])
        print(f"✅ CUDA predictions successful (shape: {predictions.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ LightGBM CUDA support failed: {e}")
        logger.warning("LightGBM CUDA support not available", error=str(e))
    

    
    # Method 2: Test CPU fallback
    print("\n🖥️  Testing CPU fallback...")
    try:
        params_cpu = {
            'device': 'cpu',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_boost_round': 1,
            'verbose': -1
        }
        
        bst_cpu = lgb.train(params_cpu, data)
        print("✅ LightGBM CPU training successful")
        logger.info("LightGBM CPU fallback verified", device="cpu")
        
        # Test prediction
        predictions = bst_cpu.predict(X[:5])
        print(f"✅ CPU predictions successful (shape: {predictions.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ LightGBM CPU training also failed: {e}")
        logger.error("LightGBM CPU training failed", error=str(e))
        return False

def test_lightgbm_performance():
    """Test LightGBM performance comparison."""
    # This is a standalone script, not a pytest test
    logger = structlog.get_logger(__name__)
    
    print("\n⚡ Performance test...")
    try:
        import lightgbm as lgb
        import numpy as np
        import time
        
        # Create larger dataset for performance test
        X = np.random.rand(1000, 20)
        y = np.random.randint(0, 2, 1000)
        data = lgb.Dataset(X, label=y)
        
        # Test CUDA performance
        try:
            params_cuda = {
                'device': 'cuda',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_boost_round': 10,
                'verbose': -1
            }
            
            start_time = time.time()
            bst_cuda = lgb.train(params_cuda, data)
            cuda_time = time.time() - start_time
            
            print(f"✅ CUDA training time: {cuda_time:.2f} seconds")
            logger.info("CUDA training performance", time_seconds=cuda_time)
            
        except Exception as e:
            print(f"⚠️  CUDA performance test failed: {e}")
            cuda_time = None
        

        
        # Test CPU performance
        try:
            params_cpu = {
                'device': 'cpu',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_boost_round': 10,
                'verbose': -1
            }
            
            start_time = time.time()
            bst_cpu = lgb.train(params_cpu, data)
            cpu_time = time.time() - start_time
            
            print(f"✅ CPU training time: {cpu_time:.2f} seconds")
            logger.info("CPU training performance", time_seconds=cpu_time)
            
            # Compare performance
            if cuda_time and cpu_time:
                speedup = cpu_time / cuda_time
                print(f"🚀 CUDA speedup: {speedup:.2f}x faster")
                logger.info("CUDA performance comparison", cuda_time=cuda_time, cpu_time=cpu_time, speedup=speedup)

            
        except Exception as e:
            print(f"❌ CPU performance test failed: {e}")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        logger.error("Performance test failed", error=str(e))

def main():
    """Main function."""
    setup_logging()
    logger = structlog.get_logger(__name__)
    
    print("🧠 LightGBM GPU Installation Verification")
    print("=" * 60)
    
    # Run tests
    success = test_lightgbm_installation()
    
    if success:
        test_lightgbm_performance()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ LightGBM installation test passed")
        logger.info("LightGBM test completed successfully")
    else:
        print("❌ LightGBM installation test failed")
        logger.error("LightGBM test failed")
    
    print("\n💡 Next steps:")
    print("   - If CUDA support failed, check CUDA installation")
    print("   - If CPU fallback works, CUDA compilation may have failed")
    print("   - Run: bash scripts/setup/lightgbm.sh install to reinstall with CUDA support")
    print("   - For GPU: Install CUDA drivers and development packages")

if __name__ == "__main__":
    main() 