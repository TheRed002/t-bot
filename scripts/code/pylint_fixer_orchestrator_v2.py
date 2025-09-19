#!/usr/bin/env python3
"""
PylintFixerOrchestrator V2 - Orchestrate the iterative pylint fixing process
Fixed version with better error handling and path management
"""

import subprocess
import time
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from pylint_error_parser import PylintErrorParser, PylintError
from prompt_builder import PromptBuilder
from claude_executor import ClaudeExecutor


class PylintFixerOrchestratorV2:
    """Orchestrate the iterative process of fixing pylint errors with better safety"""
    
    def __init__(self, 
                 max_iterations: int = 10,
                 batch_size: int = 10,
                 batch_strategy: str = 'file',
                 model: str = "claude-sonnet-4-20250514",
                 project_root: Optional[Path] = None):
        """
        Initialize the orchestrator with safety improvements
        
        Args:
            max_iterations: Maximum number of fix iterations
            batch_size: Number of errors to fix in each batch
            batch_strategy: Strategy for batching errors ('file', 'type', 'sequential')
            model: Claude model to use
            project_root: Root directory of the project (auto-detected if not provided)
        """
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.parser = PylintErrorParser()
        
        # Auto-detect project root if not provided
        if project_root is None:
            project_root = self._find_project_root()
        self.project_root = project_root
        
        # Pass project root to prompt builder
        self.prompt_builder = PromptBuilder(project_root=self.project_root)
        self.claude = ClaudeExecutor(model=model)
        self.iteration_history: List[Dict] = []
        
    def _find_project_root(self) -> Path:
        """
        Find the project root by looking for src/ directory
        
        Returns:
            Path to project root
        """
        current = Path.cwd()
        
        # Look for src/ directory in current or parent directories
        for parent in [current] + list(current.parents):
            if (parent / "src").exists() and (parent / "src").is_dir():
                return parent
        
        # Fallback to current directory
        print(f"⚠️ Could not find project root with src/ directory, using: {current}")
        return current
        
    def run_pylint(self, module_name: str, recursive: bool = True) -> Tuple[bool, List[PylintError]]:
        """
        Run pylint on a module and parse errors
        
        Args:
            module_name: Name of the module to check
            recursive: Whether to scan subdirectories (default: True)
            
        Returns:
            Tuple of (success, list of errors)
        """
        # Handle both absolute and relative paths
        if Path(module_name).is_absolute():
            module_path = Path(module_name)
            module_name = module_path.name
        else:
            module_path = self.project_root / "src" / module_name
        
        log_file = module_path / f"{module_name}_pylint.logs"
        
        # Check if module exists
        if not module_path.exists():
            print(f"❌ Module '{module_path}' does not exist")
            return False, []
        
        # Determine what to scan
        target_files = []
        if module_path.is_dir():
            # Find all Python files (recursive or not)
            if recursive:
                target_files = list(module_path.rglob("*.py"))
                if not target_files:
                    target_files = list(module_path.glob("*.py"))
            else:
                target_files = list(module_path.glob("*.py"))
            
            if not target_files:
                print(f"⚠️ No Python files found in {module_path}")
                return True, []
            
            print(f"📂 Found {len(target_files)} Python files to scan")
            # Convert to string paths
            target_files = [str(f) for f in target_files]
        else:
            # Single file
            if module_path.suffix != '.py':
                print(f"❌ {module_path} is not a Python file")
                return False, []
            target_files = [str(module_path)]
        
        # Limit files to prevent command line too long
        if len(target_files) > 100:
            print(f"⚠️ Too many files ({len(target_files)}), limiting to first 100")
            target_files = target_files[:100]
        
        # Run pylint on all target files
        cmd = [
            "pylint",
            "--disable=all",
            "--enable=E"
        ] + target_files
        
        print(f"🔍 Running pylint on {module_path}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300  # 5 minute timeout for pylint
            )
            
            # Create log directory if needed
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save output to log file
            with open(log_file, 'w') as f:
                f.write(f"Pylint Error Report for {module_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result.stdout)
            
            # Parse errors
            errors = self.parser.parse_content(result.stdout)
            
            return True, errors
            
        except subprocess.TimeoutExpired:
            print(f"❌ Pylint timed out after 5 minutes")
            return False, []
        except Exception as e:
            print(f"❌ Error running pylint: {e}")
            return False, []
    
    def fix_module(self, module_name: str, dry_run: bool = False) -> bool:
        """
        Fix all pylint errors in a module through iterative process
        
        Args:
            module_name: Name of the module to fix
            dry_run: If True, don't actually call Claude, just simulate
            
        Returns:
            True if all errors were fixed, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting pylint fix for module: {module_name}")
        if dry_run:
            print(f"🔸 DRY RUN MODE - No actual fixes will be applied")
        print(f"{'='*80}\n")
        
        # Track if we're making progress
        previous_error_count = float('inf')
        no_progress_iterations = 0
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n📍 Iteration {iteration}/{self.max_iterations}")
            print(f"{'-'*40}")
            
            # Run pylint and get current errors
            success, errors = self.run_pylint(module_name)
            
            if not success:
                print(f"❌ Failed to run pylint")
                return False
            
            # Check if we're done
            if not errors:
                print(f"✅ All errors fixed! Module {module_name} is clean.")
                self.save_summary(module_name, iteration)
                return True
            
            # Get error summary
            summary = self.parser.get_summary()
            print(f"📊 Found {summary['total_errors']} errors in {summary['files_affected']} files")
            
            # Check for progress
            current_error_count = len(errors)
            if current_error_count >= previous_error_count:
                no_progress_iterations += 1
                print(f"⚠️ No progress for {no_progress_iterations} iteration(s)")
                
                if no_progress_iterations >= 3:
                    print(f"❌ No progress for 3 iterations, stopping")
                    self.show_remaining_errors(errors)
                    return False
            else:
                no_progress_iterations = 0
                reduction = previous_error_count - current_error_count if previous_error_count != float('inf') else 0
                if reduction > 0:
                    print(f"📉 Fixed {reduction} errors in previous iteration")
            
            previous_error_count = current_error_count
            
            # Record iteration stats
            self.iteration_history.append({
                'iteration': iteration,
                'total_errors': len(errors),
                'error_types': summary['error_types']
            })
            
            # Don't try to fix if in dry run mode
            if dry_run:
                print(f"🔸 Dry run - skipping fixes")
                continue
            
            # Create batches of errors to fix
            batches = self.parser.batch_errors(
                batch_size=self.batch_size,
                strategy=self.batch_strategy
            )
            
            print(f"📦 Created {len(batches)} batch(es) for processing")
            
            # Limit batches per iteration to prevent overwhelming
            max_batches_per_iteration = 5
            if len(batches) > max_batches_per_iteration:
                print(f"⚠️ Limiting to {max_batches_per_iteration} batches this iteration")
                batches = batches[:max_batches_per_iteration]
            
            # Process each batch
            for batch_idx, batch in enumerate(batches, 1):
                print(f"\n  🔧 Processing batch {batch_idx}/{len(batches)} ({len(batch)} errors)")
                
                # Build prompt for this batch
                prompt = self.prompt_builder.build_error_fix_prompt(
                    errors=batch,
                    module_name=module_name,
                    batch_number=batch_idx,
                    total_batches=len(batches)
                )
                
                # Sanity check prompt size
                if len(prompt) > 50000:
                    print(f"  ⚠️ Prompt too large ({len(prompt)} chars), skipping batch")
                    continue
                
                # Execute fix through Claude
                try:
                    fix_success, output = self.claude.execute_prompt(
                        prompt=prompt,
                        description=f"Fixing batch {batch_idx}/{len(batches)} for {module_name}"
                    )
                    
                    if not fix_success:
                        print(f"  ❌ Failed to fix batch {batch_idx}")
                        # Continue with next batch even if one fails
                        continue
                    
                    print(f"  ✅ Batch {batch_idx} processed")
                    
                except Exception as e:
                    print(f"  ❌ Error processing batch {batch_idx}: {e}")
                    continue
                
                # Small delay between batches to avoid overwhelming
                if batch_idx < len(batches):
                    time.sleep(2)
        
        # Max iterations reached
        print(f"\n⚠️ Reached maximum iterations ({self.max_iterations})")
        _, final_errors = self.run_pylint(module_name)
        
        if final_errors:
            print(f"❌ {len(final_errors)} errors remain unfixed")
            self.show_remaining_errors(final_errors)
            return False
        
        return True
    
    def show_remaining_errors(self, errors: List[PylintError], limit: int = 10):
        """
        Display remaining errors for manual review
        
        Args:
            errors: List of remaining errors
            limit: Maximum number of errors to show
        """
        print(f"\n📋 Remaining errors (showing up to {limit}):")
        for i, error in enumerate(errors[:limit], 1):
            print(f"  {i}. {error}")
        
        if len(errors) > limit:
            print(f"  ... and {len(errors) - limit} more")
    
    def save_summary(self, module_name: str, iterations_used: int):
        """
        Save a summary of the fixing process
        
        Args:
            module_name: Name of the module
            iterations_used: Number of iterations it took
        """
        # Handle both absolute and relative paths
        if Path(module_name).is_absolute():
            module_path = Path(module_name)
            module_name = module_path.name
        else:
            module_path = self.project_root / "src" / module_name
            
        summary_file = module_path / f"{module_name}_fix_summary.txt"
        
        # Create directory if it doesn't exist
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Pylint Fix Summary for {module_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Status: SUCCESS\n")
                f.write(f"Iterations used: {iterations_used}\n")
                f.write(f"Max iterations: {self.max_iterations}\n")
                f.write(f"Batch size: {self.batch_size}\n")
                f.write(f"Batch strategy: {self.batch_strategy}\n\n")
                
                if self.iteration_history:
                    f.write("Iteration History:\n")
                    for hist in self.iteration_history:
                        f.write(f"  Iteration {hist['iteration']}: {hist['total_errors']} errors\n")
                        for code, count in hist['error_types'].items():
                            f.write(f"    - {code}: {count}\n")
            
            print(f"📄 Summary saved to {summary_file}")
        except Exception as e:
            print(f"⚠️ Could not save summary: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the fixing process
        
        Returns:
            Dictionary with process statistics
        """
        if not self.iteration_history:
            return {}
        
        initial_errors = self.iteration_history[0]['total_errors'] if self.iteration_history else 0
        final_errors = self.iteration_history[-1]['total_errors'] if self.iteration_history else 0
        
        return {
            'iterations_used': len(self.iteration_history),
            'initial_errors': initial_errors,
            'final_errors': final_errors,
            'errors_fixed': initial_errors - final_errors,
            'success_rate': ((initial_errors - final_errors) / initial_errors * 100) if initial_errors > 0 else 0
        }


def main():
    """Main entry point with enhanced safety"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestrate pylint error fixing with Claude (V2)")
    parser.add_argument("module", help="Module name or path to fix")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum fix iterations (default: 10)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Errors per batch (default: 10)")
    parser.add_argument("--batch-strategy", choices=['file', 'type', 'sequential'],
                       default='file', help="Batching strategy (default: file)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                       help="Claude model to use")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without actually calling Claude")
    parser.add_argument("--no-recursive", action="store_true",
                       help="Don't scan subdirectories")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PylintFixerOrchestratorV2(
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        batch_strategy=args.batch_strategy,
        model=args.model
    )
    
    # Fix the module
    success = orchestrator.fix_module(args.module, dry_run=args.dry_run)
    
    # Show final statistics
    stats = orchestrator.get_statistics()
    if stats:
        print(f"\n📈 Final Statistics:")
        print(f"  Iterations used: {stats['iterations_used']}")
        print(f"  Errors fixed: {stats['errors_fixed']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())