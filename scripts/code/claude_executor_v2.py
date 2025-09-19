#!/usr/bin/env python3
"""
ClaudeExecutor V2 - Execute Claude commands with better safety and timeout protection
"""

import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Tuple, Optional


class ClaudeExecutorV2:
    """Execute Claude commands with specific flags and safety measures"""
    
    def __init__(self, 
                 model: str = "claude-sonnet-4-20250514",
                 timeout: int = 900,  # 15 minutes default
                 max_prompt_size: int = 50000):
        """
        Initialize ClaudeExecutor with safety features
        
        Args:
            model: Claude model to use
            timeout: Maximum time to wait for Claude response (seconds)
            max_prompt_size: Maximum prompt size in characters
        """
        self.model = model
        self.timeout = timeout
        self.max_prompt_size = max_prompt_size
        self.claude_binary = "claude"  # Assuming claude is in PATH
        
    def execute_prompt(self, prompt: str, description: str = "") -> Tuple[bool, str]:
        """
        Execute a Claude prompt with SDK mode and safety checks
        
        Args:
            prompt: The prompt to send to Claude
            description: Optional description of what this prompt does
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        # Safety check: prompt size
        if len(prompt) > self.max_prompt_size:
            print(f"❌ Prompt too large ({len(prompt)} chars, max {self.max_prompt_size})")
            return False, f"Prompt exceeds maximum size of {self.max_prompt_size} characters"
        
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
        print(f"⏱️  Timeout: {self.timeout}s")
        print(f"📏 Prompt size: {len(prompt)} chars")
        print(f"🚀 Running Claude...")
        
        try:
            start_time = time.time()
            
            # Execute Claude subprocess with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Completed in {elapsed:.1f}s")
            
            # Check if output is reasonable
            if not result.stdout or len(result.stdout.strip()) < 10:
                print(f"⚠️ Claude returned empty or very short response")
                return False, "Claude returned empty response"
            
            return True, result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"❌ Claude timed out after {self.timeout} seconds")
            return False, f"Claude execution timed out after {self.timeout} seconds"
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Claude command failed with exit code {e.returncode}")
            if e.stderr:
                print(f"Error output: {e.stderr[:500]}")  # Limit error output
            return False, e.stderr or f"Claude failed with exit code {e.returncode}"
            
        except FileNotFoundError:
            print(f"❌ Claude binary not found. Make sure 'claude' is in your PATH")
            return False, "Claude binary not found in PATH"
            
        except MemoryError:
            print(f"❌ Out of memory while processing Claude response")
            return False, "Out of memory"
            
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            return False, f"Unexpected error: {str(e)}"
    
    def fix_pylint_errors(self, 
                         module_name: str, 
                         log_file: Optional[str] = None,
                         project_root: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Fix pylint errors for a specific module with safety checks
        
        Args:
            module_name: Name of the module to fix
            log_file: Optional path to pylint log file
            project_root: Root directory of the project
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        # Auto-detect project root if not provided
        if project_root is None:
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                if (parent / "src").exists():
                    project_root = parent
                    break
            else:
                project_root = current
        
        # Determine log file path
        if log_file is None:
            module_path = project_root / "src" / module_name
            log_file = module_path / f"{module_name}_pylint.logs"
        
        log_path = Path(log_file)
        
        if not log_path.exists():
            print(f"❌ Pylint log file not found: {log_file}")
            print(f"💡 Run 'python scripts/code/pylint-fixer.py {module_name}' first")
            return False, f"Log file {log_file} not found"
        
        # Check file size to prevent reading huge files
        file_size = log_path.stat().st_size
        if file_size > 10_000_000:  # 10MB limit
            print(f"❌ Log file too large ({file_size / 1_000_000:.1f}MB)")
            return False, "Log file too large"
        
        try:
            # Read the pylint errors
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                pylint_errors = f.read()
        except Exception as e:
            print(f"❌ Could not read log file: {e}")
            return False, f"Could not read log file: {e}"
        
        # Count errors
        error_lines = [line for line in pylint_errors.split('\n') if ': E' in line]
        error_count = len(error_lines)
        
        print(f"📊 Found {error_count} pylint errors in {module_name}")
        
        if error_count == 0:
            print(f"✅ No errors to fix in {module_name}")
            return True, "No errors found"
        
        if error_count > 100:
            print(f"⚠️ Too many errors ({error_count}), will process in batches")
        
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
{pylint_errors[:20000]}  # Limit to prevent huge prompts

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
    
    parser = argparse.ArgumentParser(description="Execute Claude to fix pylint errors (V2)")
    parser.add_argument("module", help="Module name to fix")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", 
                       help="Claude model to use")
    parser.add_argument("--log-file", help="Path to pylint log file (optional)")
    parser.add_argument("--timeout", type=int, default=900,
                       help="Timeout in seconds (default: 900)")
    
    args = parser.parse_args()
    
    # Create executor
    executor = ClaudeExecutorV2(model=args.model, timeout=args.timeout)
    
    # Fix pylint errors
    success, output = executor.fix_pylint_errors(args.module, args.log_file)
    
    if success:
        print("✅ Successfully processed module")
        if len(output) < 1000:
            print("\nOutput:")
            print(output)
        else:
            print(f"\nOutput: {len(output)} characters")
    else:
        print("❌ Failed to process module")
        sys.exit(1)


if __name__ == "__main__":
    main()