#!/usr/bin/env python3
"""
PylintFixerOrchestrator - Orchestrate the iterative pylint fixing process
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from pylint_error_parser import PylintErrorParser, PylintError
from prompt_builder import PromptBuilder
from claude_executor import ClaudeExecutor


class PylintFixerOrchestrator:
    """Orchestrate the iterative process of fixing pylint errors"""
    
    def __init__(self, 
                 max_iterations: int = 10,
                 batch_size: int = 10,
                 batch_strategy: str = 'file',
                 model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the orchestrator
        
        Args:
            max_iterations: Maximum number of fix iterations
            batch_size: Number of errors to fix in each batch
            batch_strategy: Strategy for batching errors ('file', 'type', 'sequential')
            model: Claude model to use
        """
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.parser = PylintErrorParser()
        self.prompt_builder = PromptBuilder()
        self.claude = ClaudeExecutor(model=model)
        self.iteration_history: List[Dict] = []
        
    def run_pylint(self, module_name: str) -> Tuple[bool, List[PylintError]]:
        """
        Run pylint on a module and parse errors
        
        Args:
            module_name: Name of the module to check
            
        Returns:
            Tuple of (success, list of errors)
        """
        module_path = f"src/{module_name}"
        log_file = f"src/{module_name}/{module_name}_pylint.logs"
        
        # Check if module exists
        if not Path(module_path).exists():
            print(f"❌ Module '{module_path}' does not exist")
            return False, []
        
        # Determine what to scan - if it's a directory with .py files, scan them individually
        target_files = []
        module_dir = Path(module_path)
        if module_dir.is_dir():
            # Find all Python files in the module
            target_files = list(module_dir.glob("*.py"))
            if not target_files:
                print(f"⚠️ No Python files found in {module_path}")
                return True, []
            # Convert to string paths
            target_files = [str(f) for f in target_files]
        else:
            target_files = [module_path]
        
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
                check=False
            )
            
            # Save output to log file
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f"Pylint Error Report for {module_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result.stdout)
            
            # Parse errors
            errors = self.parser.parse_content(result.stdout)
            
            return True, errors
            
        except Exception as e:
            print(f"❌ Error running pylint: {e}")
            return False, []
    
    def fix_module(self, module_name: str) -> bool:
        """
        Fix all pylint errors in a module through iterative process
        
        Args:
            module_name: Name of the module to fix
            
        Returns:
            True if all errors were fixed, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting pylint fix for module: {module_name}")
        print(f"{'='*80}\n")
        
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
            
            # Record iteration stats
            self.iteration_history.append({
                'iteration': iteration,
                'total_errors': len(errors),
                'error_types': summary['error_types']
            })
            
            # Create batches of errors to fix
            batches = self.parser.batch_errors(
                batch_size=self.batch_size,
                strategy=self.batch_strategy
            )
            
            print(f"📦 Created {len(batches)} batch(es) for processing")
            
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
                
                # Execute fix through Claude
                fix_success, output = self.claude.execute_prompt(
                    prompt=prompt,
                    description=f"Fixing batch {batch_idx}/{len(batches)} for {module_name}"
                )
                
                if not fix_success:
                    print(f"  ❌ Failed to fix batch {batch_idx}")
                    # Continue with next batch even if one fails
                    continue
                
                print(f"  ✅ Batch {batch_idx} processed")
                
                # Small delay between batches to avoid overwhelming
                if batch_idx < len(batches):
                    time.sleep(2)
            
            # Check progress after this iteration
            _, remaining_errors = self.run_pylint(module_name)
            
            if len(remaining_errors) == len(errors):
                print(f"\n⚠️ No progress made in iteration {iteration}")
                print("Errors might require manual intervention")
                self.show_remaining_errors(remaining_errors)
                return False
            
            reduction = len(errors) - len(remaining_errors)
            print(f"\n📉 Fixed {reduction} errors in this iteration")
        
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
        summary_file = f"src/{module_name}/{module_name}_fix_summary.txt"
        
        # Create directory if it doesn't exist
        Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
        
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
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestrate pylint error fixing with Claude")
    parser.add_argument("module", help="Module name to fix")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum fix iterations (default: 10)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Errors per batch (default: 10)")
    parser.add_argument("--batch-strategy", choices=['file', 'type', 'sequential'],
                       default='file', help="Batching strategy (default: file)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                       help="Claude model to use")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PylintFixerOrchestrator(
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        batch_strategy=args.batch_strategy,
        model=args.model
    )
    
    # Fix the module
    success = orchestrator.fix_module(args.module)
    
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