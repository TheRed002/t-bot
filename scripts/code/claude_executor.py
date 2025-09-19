#!/usr/bin/env python3
"""
ClaudeExecutor - Simple class to execute Claude commands for fixing pylint issues
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Optional


class ClaudeExecutor:
    """Execute Claude commands with specific flags for fixing code issues"""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize ClaudeExecutor with specific model
        
        Args:
            model: Claude model to use (default: claude-sonnet-4-20250514)
        """
        self.model = model
        self.claude_binary = "claude"  # Assuming claude is in PATH
        
    def execute_prompt(self, prompt: str, description: str = "") -> Tuple[bool, str]:
        """
        Execute a Claude prompt with SDK mode
        
        Args:
            prompt: The prompt to send to Claude
            description: Optional description of what this prompt does
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        # Build Claude command with required flags
        cmd = [
            self.claude_binary,
            "--dangerously-skip-permissions",
            "--model", self.model,
            "-p",  # SDK mode
            prompt
        ]
        
        if description:
            print(f"🤖 Executing: {description}")
        
        print(f"📋 Using model: {self.model}")
        print(f"🚀 Running Claude...")
        
        try:
            start_time = time.time()
            
            # Execute Claude subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=None  # No timeout - let Claude run as long as needed
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Completed in {elapsed:.1f}s")
            
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Claude command failed with exit code {e.returncode}")
            print(f"Error output: {e.stderr}")
            return False, e.stderr
            
        except FileNotFoundError:
            print(f"❌ Claude binary not found. Make sure 'claude' is in your PATH")
            return False, "Claude binary not found"
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False, str(e)
    
    def fix_pylint_errors(self, module_name: str, log_file: Optional[str] = None) -> Tuple[bool, str]:
        """
        Fix pylint errors for a specific module
        
        Args:
            module_name: Name of the module to fix
            log_file: Optional path to pylint log file. If not provided, will look for it in module directory
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        # Determine log file path
        if log_file is None:
            log_file = f"src/{module_name}/{module_name}_pylint.logs"
        
        log_path = Path(log_file)
        
        if not log_path.exists():
            print(f"❌ Pylint log file not found: {log_file}")
            print(f"💡 Run 'python scripts/code/pylint-fixer.py {module_name}' first")
            return False, f"Log file {log_file} not found"
        
        # Read the pylint errors
        with open(log_path, 'r') as f:
            pylint_errors = f.read()
        
        # Count errors
        error_lines = [line for line in pylint_errors.split('\n') if ': E' in line]
        error_count = len(error_lines)
        
        print(f"📊 Found {error_count} pylint errors in {module_name}")
        
        if error_count == 0:
            print(f"✅ No errors to fix in {module_name}")
            return True, "No errors found"
        
        # Create prompt for Claude to fix the errors
        prompt = f"""Fix the following pylint errors in the {module_name} module.

IMPORTANT RULES:
1. Fix ONLY the specific errors listed below
2. DO NOT refactor or change any logic
3. Preserve all existing functionality
4. Make minimal changes - only what's needed to fix the errors
5. Use proper Python typing and imports
6. This is a financial trading system - maintain all precision and safety

PYLINT ERRORS TO FIX:
{pylint_errors}

MODULE PATH: src/{module_name}

Fix these errors by:
- Adding missing imports
- Fixing undefined variables
- Correcting attribute access
- Adding proper type hints where needed
- Fixing any syntax issues

Make the minimal changes needed to resolve these specific pylint errors."""
        
        # Execute the prompt
        return self.execute_prompt(prompt, f"Fixing {error_count} pylint errors in {module_name}")


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute Claude to fix pylint errors")
    parser.add_argument("module", help="Module name to fix")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", 
                       help="Claude model to use")
    parser.add_argument("--log-file", help="Path to pylint log file (optional)")
    
    args = parser.parse_args()
    
    # Create executor
    executor = ClaudeExecutor(model=args.model)
    
    # Fix pylint errors
    success, output = executor.fix_pylint_errors(args.module, args.log_file)
    
    if success:
        print("✅ Successfully processed module")
        print("\nOutput:")
        print(output)
    else:
        print("❌ Failed to process module")
        sys.exit(1)


if __name__ == "__main__":
    main()