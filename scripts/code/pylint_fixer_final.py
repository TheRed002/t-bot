#!/usr/bin/env python3
"""
PRODUCTION-READY Pylint Fixer for T-Bot Trading System
Iteratively fixes pylint errors using Claude with comprehensive safety features
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pylint_fixer_orchestrator_v2 import PylintFixerOrchestratorV2
from claude_executor_v2 import ClaudeExecutorV2


def validate_module_name(module_name: str) -> bool:
    """
    Validate module name for safety
    
    Args:
        module_name: Module name to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Block path traversal attempts
    dangerous_patterns = ['..', '~', '$', '|', ';', '&', '>', '<', '`', '\n', '\r']
    for pattern in dangerous_patterns:
        if pattern in module_name:
            return False
    
    # Block absolute paths to system directories
    if module_name.startswith('/etc') or module_name.startswith('/usr') or module_name.startswith('/bin'):
        return False
    
    # Block Windows system paths
    if 'windows' in module_name.lower() or 'system32' in module_name.lower():
        return False
    
    return True


def find_project_root() -> Optional[Path]:
    """
    Find the project root directory
    
    Returns:
        Path to project root or None
    """
    current = Path.cwd()
    
    # Look for src/ directory in current or parent directories
    for parent in [current] + list(current.parents)[:5]:  # Limit search depth
        if (parent / "src").exists() and (parent / "src").is_dir():
            # Additional validation - check for expected structure
            if (parent / "requirements.txt").exists() or (parent / "setup.py").exists():
                return parent
            # Even without requirements.txt, if we have multiple module dirs, it's likely the root
            src_contents = list((parent / "src").iterdir())
            if len([d for d in src_contents if d.is_dir()]) >= 3:
                return parent
    
    return None


def main():
    """Main entry point with comprehensive safety and features"""
    
    parser = argparse.ArgumentParser(
        description="Fix pylint errors in T-Bot modules using Claude",
        epilog="Example: python pylint_fixer_final.py core --fix"
    )
    
    parser.add_argument(
        "module",
        help="Module name to check/fix (e.g., 'core', 'utils', 'exchanges')"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Actually fix errors using Claude (without this, just reports errors)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum fix iterations (default: 10)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of errors per batch (default: 10)"
    )
    
    parser.add_argument(
        "--batch-strategy",
        choices=['file', 'type', 'sequential'],
        default='file',
        help="How to batch errors (default: file)"
    )
    
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout for Claude in seconds (default: 900)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without actually calling Claude"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't scan subdirectories"
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        help="Explicitly set project root directory"
    )
    
    args = parser.parse_args()
    
    # Validate module name for safety
    if not validate_module_name(args.module):
        print(f"❌ Invalid module name: {args.module}")
        print("Module names cannot contain path traversal characters or system paths")
        return 1
    
    # Find or validate project root
    if args.project_root:
        project_root = Path(args.project_root)
        if not project_root.exists() or not (project_root / "src").exists():
            print(f"❌ Invalid project root: {args.project_root}")
            print("Project root must exist and contain a 'src' directory")
            return 1
    else:
        project_root = find_project_root()
        if project_root is None:
            print("❌ Could not find project root directory")
            print("Please run from the project directory or use --project-root")
            return 1
    
    print(f"📁 Project root: {project_root}")
    
    # Check if module exists
    module_path = project_root / "src" / args.module
    if not module_path.exists():
        print(f"❌ Module not found: {module_path}")
        print(f"Available modules:")
        src_dir = project_root / "src"
        for item in sorted(src_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('_'):
                print(f"  - {item.name}")
        return 1
    
    # Create orchestrator with safety features
    try:
        orchestrator = PylintFixerOrchestratorV2(
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            batch_strategy=args.batch_strategy,
            model=args.model,
            project_root=project_root
        )
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return 1
    
    # Run initial check
    print(f"\n{'='*80}")
    print(f"🔍 Checking module: {args.module}")
    print(f"{'='*80}\n")
    
    success, errors = orchestrator.run_pylint(
        args.module,
        recursive=not args.no_recursive
    )
    
    if not success:
        print(f"❌ Failed to run pylint")
        return 1
    
    if not errors:
        print(f"✅ No errors found in {args.module}!")
        return 0
    
    # Show error summary
    summary = orchestrator.parser.get_summary()
    print(f"\n📊 Error Summary:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Files affected: {summary['files_affected']}")
    print(f"  Most common: {summary['most_common_error'][0]} ({summary['most_common_error'][1]} occurrences)")
    
    print(f"\n📈 Error Distribution:")
    for code, count in sorted(summary['error_types'].items()):
        print(f"  {code}: {count}")
    
    # If not fixing, just report
    if not args.fix:
        print(f"\n💡 To fix these errors, run with --fix flag:")
        print(f"   python {Path(__file__).name} {args.module} --fix")
        return 0
    
    # Confirm before fixing
    if not args.dry_run:
        print(f"\n⚠️  About to fix {summary['total_errors']} errors in {args.module}")
        print(f"This will modify files in: {module_path}")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return 0
    
    # Fix the module
    print(f"\n🚀 Starting fix process...")
    
    try:
        fix_success = orchestrator.fix_module(
            args.module,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during fix: {e}")
        return 1
    
    # Show final statistics
    stats = orchestrator.get_statistics()
    if stats:
        print(f"\n📈 Final Statistics:")
        print(f"  Iterations used: {stats['iterations_used']}")
        print(f"  Initial errors: {stats['initial_errors']}")
        print(f"  Final errors: {stats['final_errors']}")
        print(f"  Errors fixed: {stats['errors_fixed']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
    
    if fix_success:
        print(f"\n✅ Successfully fixed all errors in {args.module}!")
        return 0
    else:
        print(f"\n⚠️ Some errors remain in {args.module}")
        print("You may need to fix these manually or run again with different settings")
        return 1


if __name__ == "__main__":
    sys.exit(main())