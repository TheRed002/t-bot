#!/usr/bin/env python3
"""Test script to reproduce the AUTO.py Claude CLI error"""

import subprocess
import time
import sys

def test_claude_call():
    """Test the exact same subprocess call that AUTO.py makes"""
    
    # Small test prompt
    prompt = "Test prompt for debugging"
    
    cmd = [
        "claude",
        "--dangerously-skip-permissions", 
        "--model", "claude-opus-4-20250514",
        "-p", prompt
    ]
    
    print(f"Testing command: {' '.join(cmd)}")
    print(f"Command list: {cmd}")
    print(f"Prompt length: {len(prompt)}")
    
    start_time = time.time()
    
    try:
        print("\nExecuting subprocess.run()...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5  # Short timeout for testing
        )
        elapsed = time.time() - start_time
        print(f"✅ Success after {elapsed:.1f}s")
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout) if result.stdout else 0}")
        print(f"Stderr length: {len(result.stderr) if result.stderr else 0}")
        
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start_time
        print(f"⏱️ Timeout after {elapsed:.1f}s")
        print(f"Partial stdout: {e.stdout[:100] if e.stdout else 'None'}")
        print(f"Partial stderr: {e.stderr[:100] if e.stderr else 'None'}")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ CalledProcessError after {elapsed:.1f}s")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout[:200] if e.stdout else 'None'}")
        print(f"Stderr: {e.stderr[:200] if e.stderr else 'None'}")
        print(f"Exception str: {str(e)}")
        print(f"Exception repr: {repr(e)}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⚠️ Unexpected exception after {elapsed:.1f}s")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception str: {str(e)}")
        print(f"Exception repr: {repr(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_claude_call()