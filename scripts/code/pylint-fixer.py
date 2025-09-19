#!/usr/bin/env python3
"""
Pylint error fixer script with iterative Claude-based fixing.
Runs pylint, detects errors, and uses Claude to fix them iteratively.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_pylint_only(module_name: str) -> None:
    """
    Run pylint on a module with only errors enabled and save output to a log file.
    
    Args:
        module_name: Name of the module to check (e.g., 'core', 'utils', 'exchanges')
    """
    # Construct the module path
    module_path = f"src/{module_name}"
    
    # Check if module exists
    if not Path(module_path).exists():
        print(f"Error: Module '{module_path}' does not exist")
        sys.exit(1)
    
    # Construct log file name - save in the module's directory
    log_file = f"src/{module_name}/{module_name}_pylint.logs"
    
    # Construct pylint command
    cmd = [
        "pylint",
        "--disable=all",
        "--enable=E",
        module_path
    ]
    
    print(f"Running pylint on {module_path}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output will be saved to: {log_file}")
    
    try:
        # Run pylint and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero return code
        )
        
        # Write output to log file (only stdout, not stderr)
        with open(log_file, 'w') as f:
            f.write(f"Pylint Error Report for {module_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.stdout)
        
        print(f"✓ Output saved to {log_file}")
        
        # Print summary
        if result.returncode == 0:
            print("✓ No errors found!")
        else:
            print(f"⚠ Pylint found issues (exit code: {result.returncode})")
            
    except FileNotFoundError:
        print("Error: pylint is not installed. Install it with: pip install pylint")
        sys.exit(1)
    except Exception as e:
        print(f"Error running pylint: {e}")
        sys.exit(1)


def main():
    """Main entry point with enhanced functionality."""
    parser = argparse.ArgumentParser(
        description="Pylint error checker and fixer",
        epilog="Example: python scripts/code/pylint-fixer.py core --fix"
    )
    
    parser.add_argument(
        "module",
        help="Module name to check/fix (e.g., 'core', 'utils')"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix errors using Claude (requires claude CLI)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum fix iterations when using --fix (default: 10)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of errors per batch when using --fix (default: 10)"
    )
    
    parser.add_argument(
        "--batch-strategy",
        choices=['file', 'type', 'sequential'],
        default='file',
        help="Batching strategy for fixes (default: file)"
    )
    
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use for fixing (default: claude-sonnet-4-20250514)"
    )
    
    args = parser.parse_args()
    
    if args.fix:
        # Import orchestrator only when needed
        try:
            from pylint_fixer_orchestrator import PylintFixerOrchestrator
            
            print(f"🤖 Starting automated fix process for {args.module}")
            
            orchestrator = PylintFixerOrchestrator(
                max_iterations=args.max_iterations,
                batch_size=args.batch_size,
                batch_strategy=args.batch_strategy,
                model=args.model
            )
            
            success = orchestrator.fix_module(args.module)
            sys.exit(0 if success else 1)
            
        except ImportError as e:
            print(f"❌ Failed to import orchestrator: {e}")
            print("Make sure all required modules are in the same directory:")
            print("  - pylint_fixer_orchestrator.py")
            print("  - pylint_error_parser.py")
            print("  - prompt_builder.py")
            print("  - claude_executor.py")
            sys.exit(1)
            
        except Exception as e:
            print(f"❌ Error during fix process: {e}")
            sys.exit(1)
    else:
        # Just run pylint check
        run_pylint_only(args.module)


if __name__ == "__main__":
    main()