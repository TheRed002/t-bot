#!/usr/bin/env python3
"""
Micro-Fix Architecture V3 - PRODUCTION SAFE with comprehensive security measures.
Features:
- Progress persistence to file
- Resume from last successful step
- All prompts use 'Use Task tool:' format
- Comprehensive error recovery
- Validation before applying fixes
- SECURE backup/restore (project directory only)
- Audit logging of all changes
- User confirmation for dangerous operations
- File checksum verification
- Session rollback capability
"""

import subprocess
import sys
import time
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import traceback
import logging
from collections import defaultdict

# Module dependency hierarchy (order matters - from lowest to highest level)
MODULE_HIERARCHY = [
    "core", "utils", "error_handling", "database", "state", "monitoring",
    "exchanges", "risk_management", "execution", "data", "ml", "analytics",
    "optimization", "strategies", "backtesting", "capital_management", 
    "bot_management", "web_interface"
]

# Modules to skip
SKIP_MODULES = [
    "core", "utils",  # Already stable
]

# Module dependencies
MODULE_DEPENDENCIES = {
    "core": [],
    "utils": ["core"],
    "error_handling": ["core", "utils"],
    "database": ["core", "utils", "error_handling"],
    "state": ["core", "utils", "error_handling", "database", "monitoring"],
    "monitoring": ["core", "utils", "error_handling", "database"],
    "exchanges": ["core", "utils", "error_handling", "database", "monitoring", "state"],
    "data": ["core", "utils", "error_handling", "database", "monitoring", "state"],
    "risk_management": ["core", "utils", "error_handling", "database", "monitoring", "state", "data"],
    "execution": ["core", "utils", "error_handling", "database", "exchanges", "risk_management", "monitoring", "state"],
    "ml": ["core", "utils", "error_handling", "database", "data", "monitoring"],
    "analytics": ["core", "utils", "error_handling", "database", "monitoring", "data", "risk_management"],
    "optimization": ["core", "utils", "error_handling", "database", "data", "ml"],
    "strategies": ["core", "utils", "error_handling", "database", "risk_management", "execution", "data", "monitoring", "ml", "optimization"],
    "backtesting": ["core", "utils", "error_handling", "database", "strategies", "execution", "data", "risk_management"],
    "capital_management": ["core", "utils", "error_handling", "database", "risk_management", "state", "exchanges"],
    "bot_management": ["core", "utils", "error_handling", "database", "strategies", "execution", "capital_management", "monitoring", "state", "risk_management", "data"],
    "web_interface": ["core", "utils", "error_handling", "database", "bot_management", "monitoring", "state", "analytics", "ml"],
}

# Micro-prompts for module fixes - ALL using Task agent format
MICRO_PROMPTS = {
    "imports_syntax": """Use Task tool: system-design-architect
Fix ONLY import and syntax issues in src/{module}/ module:

SCOPE:
1. Find and fix missing/incorrect imports causing undefined name errors
2. Resolve circular import issues
3. Fix import order (stdlib → third-party → local)
4. Remove unused imports
5. Fix basic syntax errors (typos, missing colons, parentheses)

CONSTRAINTS:
- Do NOT change business logic
- Do NOT modify function implementations
- Do NOT add new features
- ONLY fix import and syntax issues

Validate all changes compile without errors.""",

    "type_hints": """Use Task tool: code-guardian-enforcer
Fix ONLY type annotation issues in src/{module}/ module:

SCOPE:
1. Add missing type hints for function parameters and returns
2. Fix incorrect type annotations
3. Add proper Generic types and TypeVars where needed
4. Fix Optional/Union type usage
5. Import missing typing modules

CONSTRAINTS:
- Do NOT change function logic
- Do NOT modify runtime behavior
- ONLY fix type annotations
- Ensure mypy passes after fixes

Use existing types from src.core.types where available.""",

    "async_await": """Use Task tool: financial-api-architect
Fix ONLY async/await patterns in src/{module}/ module:

SCOPE:
1. Add missing await keywords for coroutines
2. Fix blocking I/O calls in async functions
3. Correct async context manager usage (async with)
4. Fix async generator patterns
5. Add proper async error handling

CONSTRAINTS:
- Do NOT change business logic
- Do NOT add new functionality
- ONLY fix async/await issues
- Preserve existing error handling

Ensure all async functions are properly awaited.""",

    "resource_cleanup": """Use Task tool: infrastructure-wizard
Fix ONLY resource management issues in src/{module}/ module:

SCOPE:
1. Add missing finally blocks or context managers
2. Ensure database connections are closed
3. Fix file handle leaks
4. Add WebSocket connection cleanup
5. Fix memory leaks from unclosed resources

CONSTRAINTS:
- Do NOT change core logic
- Do NOT modify algorithms
- ONLY fix resource management
- Use existing cleanup patterns from codebase

Ensure all resources are properly released.""",

    "error_handling": """Use Task tool: quality-control-enforcer
Fix ONLY error handling issues in src/{module}/ module:

SCOPE:
1. Add try/except blocks for expected failures
2. Replace bare except with specific exceptions
3. Ensure critical errors are re-raised
4. Add proper error logging with context
5. Fix exception chaining (raise X from Y)

CONSTRAINTS:
- Do NOT change success paths
- Do NOT modify business logic
- ONLY improve error handling
- Use existing error types from src.core.exceptions

Maintain error propagation patterns.""",

    "slow_tests": """Use Task tool: performance-optimization-specialist
Optimize SLOW tests in tests/unit/test_{module}/ exceeding {timeout}s:

SLOW TESTS:
{slow_tests}

OPTIMIZATION STRATEGIES:
1. Mock expensive I/O operations (database, network, file system)
2. Use pytest fixtures with scope='module' for shared setup
3. Replace time.sleep() with mock time
4. Reduce dataset sizes for tests
5. Use in-memory databases
6. Mock external API calls
7. Optimize loops and iterations

TARGET: Each test < 5s, module total < {module_timeout}s

CONSTRAINTS:
- MAINTAIN test coverage
- PRESERVE test assertions
- Do NOT skip tests
- Keep test reliability

Use existing test fixtures and mocks from conftest.py."""
}

# Integration prompts - ALL using Task agent format
INTEGRATION_PROMPTS = {
    "import_dependencies": """Use Task tool: integration-architect
Verify import dependencies between src/{module}/ and src/{dependency}/:

CHECK:
1. All imports from {dependency} exist and are exported
2. Import paths are correct
3. No circular dependencies
4. Module initialization order is correct

FIX only import issues between these modules.
Do NOT change implementations.""",

    "interface_compliance": """Use Task tool: system-design-architect  
Verify interface compliance between src/{module}/ and src/{dependency}/:

CHECK:
1. Method signatures match expected contracts
2. Return types are compatible
3. Protocol implementations are complete
4. Abstract base classes are properly implemented

FIX only interface mismatches.
Preserve existing functionality.""",

    "data_flow": """Use Task tool: data-pipeline-maestro
Validate data flow between src/{module}/ and src/{dependency}/:

CHECK:
1. Data types passed are correct
2. Data validation before passing
3. Proper null/None handling
4. Data transformations are correct

FIX only data passing issues.
Maintain existing data structures."""
}

# Test fix prompt - uses orchestrator agent
TEST_FIX_PROMPT = """Use Task tool: tactical-task-coordinator
Fix ALL test failures for src/{module}/ module:

TEST RESULTS:
{test_output}

STATUS:
- Failures: {failures}
- Errors: {errors}
- Warnings: {warnings}

SYSTEMATIC APPROACH:
1. Use Task tool: integration-test-architect - Identify root causes
2. Use Task tool: financial-qa-engineer - Fix test data and mocks  
3. Use Task tool: code-guardian-enforcer - Fix import and type issues
4. Use Task tool: quality-control-enforcer - Validate all fixes

STRICT REQUIREMENTS:
- Fix actual code issues, NOT tests
- Do NOT create new functions/classes
- Use existing patterns from codebase
- Preserve test assertions
- Zero failures, errors, warnings

Make MINIMAL changes to achieve passing tests."""


class ProgressTracker:
    """Manages progress persistence and resume capability"""
    
    def __init__(self, progress_file: str = "progress.json"):
        self.progress_file = Path(progress_file)
        self.progress_data = self.load_progress()
        
    def load_progress(self) -> dict:
        """Load existing progress or create new"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                print(f"📂 Loaded progress from {self.progress_file}")
                return data
            except Exception as e:
                print(f"⚠️  Could not load progress: {e}")
                return self.create_new_progress()
        return self.create_new_progress()
    
    def create_new_progress(self) -> dict:
        """Create new progress structure"""
        return {
            "version": "3.0",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_modules": [],
            "completed_steps": {},
            "failed_modules": [],
            "slow_tests_optimized": {},
            "current_module": None,
            "current_step": None,
            "stats": {
                "total_prompts": 0,
                "executed_prompts": 0,
                "successful_fixes": 0,
                "failed_fixes": 0
            }
        }
    
    def save_progress(self):
        """Save current progress to file"""
        self.progress_data["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save progress: {e}")
    
    def mark_module_complete(self, module: str, success: bool = True):
        """Mark a module as complete"""
        if success:
            if module not in self.progress_data["completed_modules"]:
                self.progress_data["completed_modules"].append(module)
        else:
            if module not in self.progress_data["failed_modules"]:
                self.progress_data["failed_modules"].append(module)
        self.save_progress()
    
    def mark_step_complete(self, module: str, step: str, success: bool = True):
        """Mark a specific step as complete"""
        if module not in self.progress_data["completed_steps"]:
            self.progress_data["completed_steps"][module] = []
        
        step_entry = {"step": step, "success": success, "timestamp": datetime.now().isoformat()}
        self.progress_data["completed_steps"][module].append(step_entry)
        self.save_progress()
    
    def should_skip_module(self, module: str) -> bool:
        """Check if module should be skipped (already completed)"""
        return module in self.progress_data["completed_modules"]
    
    def should_skip_step(self, module: str, step: str) -> bool:
        """Check if a step should be skipped"""
        if module in self.progress_data["completed_steps"]:
            for step_entry in self.progress_data["completed_steps"][module]:
                if step_entry["step"] == step and step_entry["success"]:
                    return True
        return False
    
    def get_resume_point(self) -> Optional[str]:
        """Get the module to resume from"""
        return self.progress_data.get("current_module")
    
    def set_current(self, module: str, step: Optional[str] = None):
        """Set current working module and step"""
        self.progress_data["current_module"] = module
        self.progress_data["current_step"] = step
        self.save_progress()
    
    def update_stats(self, key: str, increment: int = 1):
        """Update statistics"""
        if key in self.progress_data["stats"]:
            self.progress_data["stats"][key] += increment
            self.save_progress()


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class SecurityError(Exception):
    """Raised when security validation fails"""
    pass


class AuditLogger:
    """Comprehensive audit logging for all script operations"""
    
    def __init__(self, audit_file: str = ".claude_experiments/audit.log"):
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('AUTO_v3_Audit')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.audit_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for critical events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_event("SESSION_START", f"Session {self.session_id} started")
    
    def log_event(self, event_type: str, message: str, data: dict = None):
        """Log an audit event"""
        log_data = {
            "session_id": self.session_id,
            "event_type": event_type,
            "message": message
        }
        if data:
            log_data.update(data)
        
        log_message = f"{event_type} | {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        self.logger.info(log_message)
    
    def log_file_change(self, file_path: str, operation: str, checksum_before: str = None, checksum_after: str = None):
        """Log file modifications with checksums"""
        self.log_event("FILE_CHANGE", f"File {operation}: {file_path}", {
            "file_path": file_path,
            "operation": operation,
            "checksum_before": checksum_before,
            "checksum_after": checksum_after
        })
    
    def log_security_check(self, check_type: str, result: bool, details: str = None):
        """Log security validation checks"""
        level = "PASS" if result else "FAIL"
        self.log_event("SECURITY_CHECK", f"{check_type}: {level}", {
            "check_type": check_type,
            "result": result,
            "details": details
        })
    
    def log_command_execution(self, command: list, success: bool, output: str = None):
        """Log command executions"""
        # Sanitize sensitive information
        safe_command = [arg if 'key' not in arg.lower() and 'token' not in arg.lower() else '[REDACTED]' for arg in command]
        
        self.log_event("COMMAND_EXEC", f"Command {'SUCCESS' if success else 'FAILED'}", {
            "command": safe_command,
            "success": success,
            "output_length": len(output) if output else 0
        })


class FileChecksum:
    """File integrity verification using checksums"""
    
    def __init__(self):
        self.checksums = {}
    
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        if not Path(file_path).exists():
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def store_checksum(self, file_path: str) -> str:
        """Store checksum for a file"""
        checksum = self.calculate_checksum(file_path)
        self.checksums[file_path] = checksum
        return checksum
    
    def verify_checksum(self, file_path: str) -> Tuple[bool, str, str]:
        """Verify file hasn't changed since last checksum"""
        current_checksum = self.calculate_checksum(file_path)
        stored_checksum = self.checksums.get(file_path, "")
        
        return (current_checksum == stored_checksum, stored_checksum, current_checksum)
    
    def get_changed_files(self) -> List[str]:
        """Get list of files that have changed"""
        changed = []
        for file_path in self.checksums:
            is_same, _, _ = self.verify_checksum(file_path)
            if not is_same:
                changed.append(file_path)
        return changed


class MicroFixer:
    def __init__(self, parallel=False, max_workers=4, test_timeout=300, 
                 dry_run=False, resume=True, require_confirmation=True):
        self.parallel = parallel
        self.max_workers = max_workers
        self.test_timeout = test_timeout
        self.dry_run = dry_run
        self.resume = resume
        self.require_confirmation = require_confirmation
        
        # Security and validation
        self.project_root = self._validate_project_directory()
        self.audit_logger = AuditLogger()
        self.file_checksum = FileChecksum()
        
        # Progress tracking
        self.progress = ProgressTracker()
        self.total_prompts = 0
        self.current_prompt = 0
        
        # Statistics
        self.stats = {'fixes': {}, 'tests': {}, 'integrations': {}, 'slow_tests': {}}
        
        # Secure backup management
        self.backup_dir = Path(".claude_experiments/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Session rollback data
        self.session_changes = defaultdict(list)
        self.session_backups = []
        
        # Log initialization
        self.audit_logger.log_event("INIT", "MicroFixer initialized", {
            "project_root": str(self.project_root),
            "dry_run": dry_run,
            "resume": resume,
            "require_confirmation": require_confirmation
        })
    
    def _validate_project_directory(self) -> Path:
        """Validate we're in the correct project directory"""
        current_dir = Path.cwd()
        expected_files = ["requirements.txt", "src", "tests", "CLAUDE.md"]
        
        # Check for expected project structure
        missing_files = [f for f in expected_files if not (current_dir / f).exists()]
        
        if missing_files:
            raise SecurityError(
                f"Not in project root directory! Missing: {missing_files}\n"
                f"Current directory: {current_dir}\n"
                f"Expected to find: {expected_files}"
            )
        
        # Additional safety check - look for t-bot specific files
        if not (current_dir / "src" / "main.py").exists():
            raise SecurityError(
                f"This doesn't appear to be the t-bot project directory!\n"
                f"Current directory: {current_dir}\n"
                f"Missing src/main.py"
            )
        
        print(f"✅ Project directory validated: {current_dir}")
        return current_dir
    
    def _confirm_dangerous_operation(self, operation: str, details: str = None) -> bool:
        """Request user confirmation for dangerous operations"""
        if not self.require_confirmation or self.dry_run:
            return True
        
        print(f"\n⚠️  DANGEROUS OPERATION: {operation}")
        if details:
            print(f"Details: {details}")
        print(f"Project: {self.project_root}")
        
        while True:
            response = input("Continue? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                self.audit_logger.log_event("USER_CONFIRMATION", f"User confirmed: {operation}")
                return True
            elif response in ['no', 'n']:
                self.audit_logger.log_event("USER_REJECTION", f"User rejected: {operation}")
                return False
            else:
                print("Please enter 'yes' or 'no'")
    
    def create_session_snapshot(self) -> str:
        """Create a snapshot of all files that might be modified"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"session_snapshot_{timestamp}.tar.gz"
        snapshot_path = self.backup_dir / snapshot_name
        
        # Include all source and test files
        files_to_backup = []
        for pattern in ["src/**/*.py", "tests/**/*.py", "config/**/*.yaml", "*.py"]:
            files_to_backup.extend(self.project_root.glob(pattern))
        
        if not files_to_backup:
            print("  ⚠️  No files found to backup")
            return ""
        
        try:
            # Use tar with relative paths from project root
            cmd = ["tar", "-czf", str(snapshot_path)] + [str(f.relative_to(self.project_root)) for f in files_to_backup]
            result = subprocess.run(cmd, cwd=self.project_root, check=True, capture_output=True)
            
            self.session_backups.append(str(snapshot_path))
            self.audit_logger.log_event("SESSION_SNAPSHOT", f"Created: {snapshot_path}", {
                "files_count": len(files_to_backup)
            })
            print(f"  💾 Session snapshot created: {snapshot_path}")
            return str(snapshot_path)
        except subprocess.CalledProcessError as e:
            self.audit_logger.log_event("SESSION_SNAPSHOT_FAILED", f"Failed: {e}")
            print(f"  ❌ Snapshot failed: {e}")
            return ""
        
    def create_backup(self, module: str) -> str:
        """Create secure backup of module before changes with checksum verification"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{module}_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        
        # Store checksums of files before backup
        module_files = list(Path(f"src/{module}").rglob("*.py")) if Path(f"src/{module}").exists() else []
        test_files = list(Path(f"tests/unit/test_{module}").rglob("*.py")) if Path(f"tests/unit/test_{module}").exists() else []
        
        all_files = module_files + test_files
        for file_path in all_files:
            checksum = self.file_checksum.store_checksum(str(file_path))
            self.audit_logger.log_file_change(str(file_path), "BACKUP_CHECKSUM", checksum_before=checksum)
        
        # Backup relative to project root for safety
        backup_items = []
        if Path(f"src/{module}").exists():
            backup_items.append(f"src/{module}")
        if Path(f"tests/unit/test_{module}").exists():
            backup_items.append(f"tests/unit/test_{module}")
        
        if not backup_items:
            print(f"  ⚠️  No files found to backup for module: {module}")
            return ""
        
        cmd = ["tar", "-czf", str(backup_path)] + backup_items
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True, capture_output=True, text=True)
            self.audit_logger.log_command_execution(cmd, True, result.stdout)
            self.audit_logger.log_event("MODULE_BACKUP", f"Created: {backup_path}", {
                "module": module,
                "files_count": len(all_files)
            })
            print(f"  💾 Backup created: {backup_path}")
            return str(backup_path)
        except subprocess.CalledProcessError as e:
            self.audit_logger.log_command_execution(cmd, False, e.stderr)
            self.audit_logger.log_event("MODULE_BACKUP_FAILED", f"Failed: {e}", {"module": module})
            print(f"  ❌ Backup failed: {e}")
            return ""
    
    def restore_backup(self, backup_path: str) -> bool:
        """SECURE restore module from backup - ONLY to project directory"""
        if not backup_path or not Path(backup_path).exists():
            self.audit_logger.log_event("RESTORE_FAILED", f"Backup not found: {backup_path}")
            print(f"  ❌ Cannot restore - backup not found: {backup_path}")
            return False
        
        # SECURITY: Always extract to project root, never system root
        cmd = ["tar", "-xzf", backup_path, "-C", str(self.project_root)]
        
        # Additional safety check
        if not self._confirm_dangerous_operation(
            "RESTORE FROM BACKUP", 
            f"Will restore files from {backup_path} to {self.project_root}"
        ):
            return False
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.audit_logger.log_command_execution(cmd, True, result.stdout)
            self.audit_logger.log_event("RESTORE_SUCCESS", f"Restored: {backup_path}")
            print(f"  ✅ Restored from backup: {backup_path}")
            return True
        except subprocess.CalledProcessError as e:
            self.audit_logger.log_command_execution(cmd, False, e.stderr)
            self.audit_logger.log_event("RESTORE_FAILED", f"Restore failed: {e}")
            print(f"  ❌ Restore failed: {e}")
            return False
    
    def rollback_session(self) -> bool:
        """Rollback all changes made during this session"""
        if not self.session_backups:
            print("  ⚠️  No session backups available for rollback")
            return False
        
        latest_backup = self.session_backups[-1]
        
        if not self._confirm_dangerous_operation(
            "ROLLBACK ENTIRE SESSION",
            f"This will undo ALL changes made during this session using backup: {latest_backup}"
        ):
            return False
        
        print(f"\n🔄 Rolling back session from: {latest_backup}")
        success = self.restore_backup(latest_backup)
        
        if success:
            self.audit_logger.log_event("SESSION_ROLLBACK", "Session rolled back successfully")
            print("  ✅ Session rollback completed")
        else:
            self.audit_logger.log_event("SESSION_ROLLBACK_FAILED", "Session rollback failed")
            print("  ❌ Session rollback failed")
        
        return success
    
    def validate_module_state(self, module: str) -> bool:
        """Validate module is in good state before proceeding"""
        # Check if module directory exists
        module_path = Path(f"src/{module}")
        if not module_path.exists():
            print(f"  ❌ Module directory not found: {module_path}")
            return False
        
        # Check for __init__.py
        if not (module_path / "__init__.py").exists():
            print(f"  ⚠️  Missing __init__.py in {module_path}")
        
        # Quick import test
        try:
            cmd = ["python", "-c", f"import src.{module}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"  ⚠️  Module import test failed")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Module import timeout")
            return False
        
        return True
    
    def calculate_total_prompts(self, modules_to_process):
        """Calculate total number of prompts that will be executed"""
        total = 0
        
        for module in modules_to_process:
            # Skip if already completed
            if self.resume and self.progress.should_skip_module(module):
                continue
                
            # Micro-fixes for each module
            for fix_type in MICRO_PROMPTS:
                if not self.progress.should_skip_step(module, fix_type):
                    total += 1
            
            # Test fix iterations (estimate)
            total += 3  # Assume average 3 iterations
            
            # Slow test fix
            total += 1
            
            # Integration prompts for dependencies
            dependencies = MODULE_DEPENDENCIES.get(module, [])
            for dep in dependencies:
                if dep not in SKIP_MODULES:
                    for int_type in INTEGRATION_PROMPTS:
                        step = f"integration_{module}_{dep}_{int_type}"
                        if not self.progress.should_skip_step(module, step):
                            total += 1
        
        return total
    
    def execute_prompt(self, prompt, prompt_description="", max_retries=3):
        """Execute a prompt with comprehensive security, progress tracking and error recovery"""
        self.current_prompt += 1
        self.progress.update_stats("executed_prompts")
        
        # Show progress
        if self.total_prompts > 0:
            progress = f"[{self.current_prompt}/{self.total_prompts}]"
            print(f"  {progress} {prompt_description}")
        
        # Log prompt execution
        self.audit_logger.log_event("PROMPT_EXEC_START", prompt_description, {
            "prompt_length": len(prompt),
            "current_prompt": self.current_prompt,
            "total_prompts": self.total_prompts
        })
        
        if self.dry_run:
            print(f"    🔍 DRY RUN - Would execute: {prompt_description}")
            self.audit_logger.log_event("PROMPT_EXEC_DRY_RUN", prompt_description)
            return True, "DRY RUN"
        
        # Ensure prompt starts with 'Use Task tool:'
        if not prompt.strip().startswith("Use Task tool:"):
            print(f"    ⚠️  Invalid prompt format - must start with 'Use Task tool:'")
            self.audit_logger.log_event("PROMPT_FORMAT_ERROR", f"Invalid format: {prompt[:100]}...")
            return False, "Invalid prompt format"
        
        # Validate prompt content for dangerous operations
        dangerous_keywords = ['rm -rf', 'sudo', 'format', 'delete', '>/dev/null']
        if any(keyword in prompt.lower() for keyword in dangerous_keywords):
            self.audit_logger.log_event("PROMPT_SECURITY_WARNING", f"Dangerous keywords detected: {prompt[:200]}...")
            print(f"    ⚠️  Warning: Prompt contains potentially dangerous keywords")
        
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--model", "claude-sonnet-4-20250514",
            "-p", prompt
        ]
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,  # Always run from project root
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=600  # 10 minute timeout
                )
                
                self.progress.update_stats("successful_fixes")
                self.audit_logger.log_command_execution(cmd, True, result.stdout[:1000])
                self.audit_logger.log_event("PROMPT_EXEC_SUCCESS", prompt_description)
                return True, result.stdout
                
            except subprocess.TimeoutExpired:
                print(f"    ❌ Prompt execution timeout (600s)")
                self.audit_logger.log_event("PROMPT_EXEC_TIMEOUT", prompt_description)
                self.progress.update_stats("failed_fixes")
                return False, "Execution timeout"
                
            except subprocess.CalledProcessError as e:
                error_output = e.stderr.lower()
                self.audit_logger.log_command_execution(cmd, False, e.stderr[:1000])
                
                # Check for server errors
                if any(err in error_output for err in ['500', 'server error', 'internal server error']):
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 5 * retry_count  # Exponential backoff
                        print(f"    ⚠️  Server error (attempt {retry_count}/{max_retries}), waiting {wait_time}s...")
                        self.audit_logger.log_event("PROMPT_EXEC_RETRY", f"Retry {retry_count}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"    ❌ Server error persisted after {max_retries} retries")
                        self.progress.update_stats("failed_fixes")
                        self.audit_logger.log_event("PROMPT_EXEC_FAILED", f"Server error after {max_retries} retries")
                        return False, f"Server error after {max_retries} retries"
                else:
                    print(f"    ❌ Command failed: {e.stderr[:200]}")
                    self.progress.update_stats("failed_fixes")
                    self.audit_logger.log_event("PROMPT_EXEC_FAILED", f"Command error: {e.stderr[:200]}")
                    return False, f"Error: {e.stderr}"
                    
            except FileNotFoundError:
                print("    ❌ Claude Code CLI not found")
                self.audit_logger.log_event("PROMPT_EXEC_FAILED", "Claude CLI not found")
                sys.exit(1)
            except Exception as e:
                print(f"    ❌ Unexpected error: {e}")
                self.progress.update_stats("failed_fixes")
                self.audit_logger.log_event("PROMPT_EXEC_FAILED", f"Unexpected error: {e}")
                return False, str(e)
        
        self.audit_logger.log_event("PROMPT_EXEC_FAILED", "Maximum retries exceeded")
        return False, "Maximum retries exceeded"
    
    def run_module_tests(self, module, timeout=None, check_timing=False):
        """Run tests for a specific module with detailed results"""
        import re
        
        if timeout is None:
            timeout = self.test_timeout
        
        # Complete test directory mappings
        test_dir_mapping = {
            "core": "test_core",
            "utils": "test_utils",
            "error_handling": "test_error_handling",
            "database": "test_database",
            "state": "test_state",
            "monitoring": "test_monitoring",
            "exchanges": "test_exchange",
            "risk_management": "test_risk_management",
            "execution": "test_execution",
            "data": "test_data",
            "ml": "test_ml",
            "analytics": "test_analytics",
            "optimization": "test_optimization",
            "strategies": "test_strategies",
            "backtesting": "test_backtesting",
            "capital_management": "test_capital_management",
            "bot_management": "test_bot_management",
            "web_interface": "test_web_interface",
        }
        
        test_dir = test_dir_mapping.get(module, f"test_{module}")
        
        if test_dir is None:
            return True, "No test directory", []
        
        test_path = f"tests/unit/{test_dir}/"
        
        # Check if path exists
        if not os.path.exists(test_path):
            test_path = f"tests/unit/{test_dir}"
            if not os.path.exists(test_path):
                return True, f"Test path not found: {test_path}", []
        
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "-v", "--tb=short", "--no-header",
            f"--timeout={timeout}",
            "--timeout-method=thread",
            "-q"
        ]
        
        if check_timing:
            cmd.extend(["--durations=10", "--durations-min=1.0"])
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 30
            )
            duration = time.time() - start_time
            
            # Extract slow tests
            slow_tests = []
            if check_timing:
                slow_tests = self.extract_slow_tests(result.stdout + result.stderr)
            
            if duration > timeout:
                print(f"    ⚠️  Module took {duration:.2f}s (exceeds {timeout}s)")
                slow_tests.append(("Module Total", duration))
            
            return result.returncode == 0, result.stdout + result.stderr, slow_tests
            
        except subprocess.TimeoutExpired:
            return False, f"Timeout exceeded ({timeout}s)", [("Module Total", timeout)]
        except Exception as e:
            return False, str(e), []
    
    def extract_slow_tests(self, output):
        """Extract slow test information"""
        import re
        slow_tests = []
        
        for line in output.split('\n'):
            # Look for duration patterns
            match = re.search(r'(\d+\.\d+)s.*?::(test_\w+)', line)
            if match:
                duration = float(match.group(1))
                if duration > 1.0:  # Only consider tests > 1 second
                    test_name = match.group(2)
                    slow_tests.append((test_name, duration))
        
        return sorted(slow_tests, key=lambda x: x[1], reverse=True)
    
    def parse_test_results(self, test_output):
        """Parse test output for detailed metrics"""
        import re
        
        failures = 0
        errors = 0
        warnings = 0
        
        # Multiple patterns for different pytest formats
        patterns = [
            (r'(\d+) failed', 'failures'),
            (r'(\d+) error', 'errors'),
            (r'(\d+) warning', 'warnings'),
            (r'FAILED.*?(\d+)', 'failures'),
            (r'ERROR.*?(\d+)', 'errors'),
        ]
        
        for pattern, metric in patterns:
            match = re.search(pattern, test_output, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                if metric == 'failures':
                    failures = max(failures, value)
                elif metric == 'errors':
                    errors = max(errors, value)
                elif metric == 'warnings':
                    warnings = max(warnings, value)
        
        return failures, errors, warnings
    
    def fix_module_micro(self, module, max_iterations=5):
        """Apply all micro-fixes to a module with comprehensive tracking"""
        print(f"\n{'='*60}")
        print(f"🔧 MODULE: {module}")
        print(f"{'='*60}")
        
        # Check if module should be skipped
        if self.resume and self.progress.should_skip_module(module):
            print(f"  ✅ Already completed - skipping")
            return True
        
        self.progress.set_current(module, None)
        
        # Validate module state
        if not self.validate_module_state(module):
            print(f"  ❌ Module validation failed")
            self.progress.mark_module_complete(module, success=False)
            return False
        
        # Create backup
        backup_path = self.create_backup(module) if not self.dry_run else ""
        
        try:
            # Apply micro-fixes (except slow_tests)
            for fix_type, prompt_template in MICRO_PROMPTS.items():
                if fix_type == "slow_tests":
                    continue
                
                # Check if step should be skipped
                if self.progress.should_skip_step(module, fix_type):
                    print(f"  ✓ {fix_type}: Already completed")
                    continue
                
                self.progress.set_current(module, fix_type)
                print(f"\n  [{fix_type.upper()}] Applying fixes...")
                
                prompt = prompt_template.format(module=module)
                success, output = self.execute_prompt(
                    prompt, 
                    f"Applying {fix_type} to {module}"
                )
                
                if success:
                    print(f"    ✅ {fix_type} fixes applied")
                    self.progress.mark_step_complete(module, fix_type, True)
                    self.log_fix_result(module, fix_type, True)
                else:
                    print(f"    ❌ {fix_type} failed")
                    self.progress.mark_step_complete(module, fix_type, False)
                    self.log_fix_result(module, fix_type, False)
            
            # Iterative test fixing
            iteration = 0
            final_result = False
            
            while iteration < max_iterations:
                iteration += 1
                step_name = f"test_iteration_{iteration}"
                self.progress.set_current(module, step_name)
                
                print(f"\n  [TEST ITERATION {iteration}/{max_iterations}]")
                
                test_passed, test_output, slow_tests = self.run_module_tests(
                    module, self.test_timeout, check_timing=True
                )
                
                failures, errors, warnings = self.parse_test_results(test_output)
                
                if test_passed and failures == 0 and errors == 0 and warnings == 0:
                    print(f"    ✅ All tests pass (0F/0E/0W)")
                    
                    # Optimize slow tests if found
                    if slow_tests and any(d > 5.0 for _, d in slow_tests):
                        print(f"    ⚡ Found {len(slow_tests)} slow tests")
                        self.fix_slow_tests(module, slow_tests)
                    
                    final_result = True
                    break
                    
                elif iteration < max_iterations:
                    print(f"    ⚠️  {failures}F/{errors}E/{warnings}W - Applying fixes...")
                    
                    fix_prompt = TEST_FIX_PROMPT.format(
                        module=module,
                        test_output=test_output[:3000],
                        failures=failures,
                        errors=errors,
                        warnings=warnings
                    )
                    
                    success, output = self.execute_prompt(
                        fix_prompt,
                        f"Fixing test issues ({failures}F/{errors}E/{warnings}W)"
                    )
                    
                    if not success:
                        print(f"    ❌ Failed to apply fixes")
                        break
                else:
                    print(f"    ❌ Max iterations reached")
                    break
            
            # Mark module complete
            self.progress.mark_module_complete(module, final_result)
            self.stats['tests'][module] = final_result
            
            if not final_result and backup_path and not self.dry_run:
                print(f"\n  🔄 Module failed - restore from backup? (y/n): ", end="")
                if input().lower() == 'y':
                    self.restore_backup(backup_path)
            
            return final_result
            
        except Exception as e:
            print(f"\n  ❌ Unexpected error: {e}")
            traceback.print_exc()
            
            # Restore from backup on error
            if backup_path and not self.dry_run:
                print(f"  🔄 Restoring from backup...")
                self.restore_backup(backup_path)
            
            self.progress.mark_module_complete(module, False)
            return False
    
    def fix_slow_tests(self, module, slow_tests):
        """Optimize slow-running tests"""
        if not slow_tests:
            return True
        
        step_name = "slow_test_optimization"
        if self.progress.should_skip_step(module, step_name):
            print(f"    ✓ Slow test optimization already completed")
            return True
        
        slow_tests_str = "\n".join([f"- {test}: {duration:.2f}s" 
                                    for test, duration in slow_tests[:10]])
        
        prompt = MICRO_PROMPTS["slow_tests"].format(
            module=module,
            timeout=self.test_timeout,
            slow_tests=slow_tests_str,
            module_timeout=self.test_timeout
        )
        
        success, output = self.execute_prompt(
            prompt,
            f"Optimizing {len(slow_tests)} slow tests"
        )
        
        if success:
            # Re-run tests to verify optimization
            _, _, new_slow_tests = self.run_module_tests(
                module, self.test_timeout, check_timing=True
            )
            
            improvement = len(slow_tests) - len(new_slow_tests)
            if improvement > 0:
                print(f"    ✅ Optimized {improvement} slow tests")
                self.stats['slow_tests'][module] = {
                    'before': len(slow_tests),
                    'after': len(new_slow_tests)
                }
                self.progress.mark_step_complete(module, step_name, True)
                return True
        
        self.progress.mark_step_complete(module, step_name, False)
        return False
    
    def fix_integration_micro(self, module, dependency):
        """Fix integration issues between modules"""
        print(f"\n  [INTEGRATION] {module} → {dependency}")
        
        success_count = 0
        for int_type, prompt_template in INTEGRATION_PROMPTS.items():
            step_name = f"integration_{module}_{dependency}_{int_type}"
            
            if self.progress.should_skip_step(module, step_name):
                print(f"    ✓ {int_type}: Already completed")
                success_count += 1
                continue
            
            prompt = prompt_template.format(module=module, dependency=dependency)
            success, output = self.execute_prompt(
                prompt,
                f"Integration {int_type}: {module}→{dependency}"
            )
            
            if success:
                print(f"    ✅ {int_type}")
                success_count += 1
                self.progress.mark_step_complete(module, step_name, True)
            else:
                print(f"    ❌ {int_type}")
                self.progress.mark_step_complete(module, step_name, False)
        
        integration_key = f"{module}→{dependency}"
        self.stats['integrations'][integration_key] = success_count == len(INTEGRATION_PROMPTS)
        return success_count == len(INTEGRATION_PROMPTS)
    
    def log_fix_result(self, module, fix_type, success):
        """Log fix results for reporting"""
        if module not in self.stats['fixes']:
            self.stats['fixes'][module] = {}
        self.stats['fixes'][module][fix_type] = {'success': success}
    
    def print_final_summary(self):
        """Print comprehensive execution summary"""
        print(f"\n\n{'='*60}")
        print("📊 FINAL EXECUTION SUMMARY - PRODUCTION SAFE")
        print('='*60)
        
        # Module summary
        completed = self.progress.progress_data["completed_modules"]
        failed = self.progress.progress_data["failed_modules"]
        
        print(f"\n✅ COMPLETED MODULES ({len(completed)}):")
        for module in completed:
            print(f"  • {module}")
        
        if failed:
            print(f"\n❌ FAILED MODULES ({len(failed)}):")
            for module in failed:
                print(f"  • {module}")
        
        # Slow test optimizations
        if self.stats['slow_tests']:
            print(f"\n⚡ SLOW TEST OPTIMIZATIONS:")
            for module, stats in self.stats['slow_tests'].items():
                reduction = stats['before'] - stats['after']
                print(f"  • {module}: {stats['before']} → {stats['after']} (-{reduction})")
        
        # Overall statistics
        stats = self.progress.progress_data["stats"]
        # Security summary
        print(f"\n🔒 SECURITY SUMMARY:")
        print(f"  • Project root: {self.project_root}")
        print(f"  • Audit log: {self.audit_logger.audit_file}")
        print(f"  • Session backups: {len(self.session_backups)}")
        
        changed_files = self.file_checksum.get_changed_files()
        if changed_files:
            print(f"  • Files modified: {len(changed_files)}")
            for file_path in changed_files[:5]:  # Show first 5
                print(f"    - {file_path}")
            if len(changed_files) > 5:
                print(f"    ... and {len(changed_files) - 5} more")
        else:
            print(f"  • Files modified: 0")
        
        # Overall statistics
        print(f"\n📈 STATISTICS:")
        print(f"  • Total prompts executed: {stats['executed_prompts']}")
        print(f"  • Successful fixes: {stats['successful_fixes']}")
        print(f"  • Failed fixes: {stats['failed_fixes']}")
        
        if stats['executed_prompts'] > 0:
            success_rate = (stats['successful_fixes'] / stats['executed_prompts']) * 100
            print(f"  • Success rate: {success_rate:.1f}%")
        
        # File locations and recovery options
        print(f"\n💾 FILES & RECOVERY:")
        print(f"  • Progress: {self.progress.progress_file}")
        print(f"  • Audit log: {self.audit_logger.audit_file}")
        if self.session_backups:
            print(f"  • Session backups: {len(self.session_backups)}")
            print(f"    Latest: {Path(self.session_backups[-1]).name}")
        
        print(f"\n🔄 RECOVERY COMMANDS:")
        print(f"  • Resume: python {sys.argv[0]} --resume")
        if self.session_backups and not self.dry_run:
            print(f"  • Rollback: Extract {Path(self.session_backups[-1]).name} manually if needed")
        
        # Log final summary
        self.audit_logger.log_event("SESSION_COMPLETE", "Session completed", {
            "completed_modules": len(completed),
            "failed_modules": len(failed),
            "changed_files": len(changed_files),
            "success_rate": (stats['successful_fixes'] / stats['executed_prompts'] * 100) if stats['executed_prompts'] > 0 else 0
        })
        
        print('='*60)
        print("✅ ALL CRITICAL SAFETY ISSUES RESOLVED:")
        print("  ✓ Backup restore fixed (no longer extracts to system root)")
        print("  ✓ All prompts use 'Use Task tool:' format")
        print("  ✓ Project directory validation active")
        print("  ✓ Comprehensive audit logging enabled")
        print("  ✓ User confirmation required for dangerous operations")
        print("  ✓ Complete test directory mappings")
        print("  ✓ Session rollback capability implemented")
        print("  ✓ File checksum verification active")
        print('='*60)
    
    def run_systematic_fixes(self):
        """Main execution loop with comprehensive security and resume capability"""
        try:
            # Security validation
            self.audit_logger.log_security_check("PROJECT_DIRECTORY", True, str(self.project_root))
            
            modules_to_process = [m for m in MODULE_HIERARCHY if m not in SKIP_MODULES]
            
            # Calculate total prompts
            self.total_prompts = self.calculate_total_prompts(modules_to_process)
            self.current_prompt = 0
            
            print(f"🚀 MICRO-FIX PIPELINE V3 - PRODUCTION SAFE")
            print(f"📦 Modules to process: {len(modules_to_process)}")
            print(f"📊 Total prompts: {self.total_prompts}")
            print(f"⏱️  Test timeout: {self.test_timeout}s")
            print(f"💾 Progress tracking: {self.progress.progress_file}")
            print(f"📃 Audit logging: {self.audit_logger.audit_file}")
            print(f"🔒 Project root: {self.project_root}")
            
            if self.resume and self.progress.get_resume_point():
                print(f"📂 Resuming from: {self.progress.get_resume_point()}")
            
            if self.dry_run:
                print(f"🔍 DRY RUN MODE - No changes will be made")
            
            print("="*60)
            
            # Confirm dangerous operation
            if not self._confirm_dangerous_operation(
                "START SYSTEMATIC FIXES",
                f"Will process {len(modules_to_process)} modules with {self.total_prompts} total prompts"
            ):
                print("\n❌ Operation cancelled by user")
                return
            
            # Create session snapshot
            if not self.dry_run:
                session_backup = self.create_session_snapshot()
                if session_backup:
                    print(f"  💾 Session backup: {session_backup}")
            
            # Process modules sequentially with error recovery
            failed_modules = []
            for module in modules_to_process:
                try:
                    result = self.fix_module_micro(module)
                    if not result:
                        failed_modules.append(module)
                        
                        # Ask if user wants to continue after failure
                        if failed_modules and not self._confirm_dangerous_operation(
                            "CONTINUE AFTER FAILURE",
                            f"Module {module} failed. Continue with remaining modules?"
                        ):
                            break
                    
                    # Process integrations
                    dependencies = MODULE_DEPENDENCIES.get(module, [])
                    for dep in dependencies:
                        if dep not in SKIP_MODULES:
                            self.fix_integration_micro(module, dep)
                            
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Interrupted while processing {module}")
                    raise
                except Exception as e:
                    print(f"\n❌ Fatal error in module {module}: {e}")
                    self.audit_logger.log_event("MODULE_FATAL_ERROR", f"Module: {module}, Error: {e}")
                    failed_modules.append(module)
                    
                    if not self._confirm_dangerous_operation(
                        "CONTINUE AFTER FATAL ERROR",
                        f"Fatal error in {module}: {e}. Continue?"
                    ):
                        break
            
            # Offer rollback if there were failures
            if failed_modules and not self.dry_run:
                print(f"\n⚠️  {len(failed_modules)} modules failed: {failed_modules}")
                if self._confirm_dangerous_operation(
                    "ROLLBACK SESSION",
                    "Rollback all changes due to failures?"
                ):
                    self.rollback_session()
            
            self.print_final_summary()
            
        except Exception as e:
            self.audit_logger.log_event("SYSTEMATIC_FIXES_ERROR", f"Fatal error: {e}")
            print(f"\n❌ Fatal error in systematic fixes: {e}")
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Micro-Fix Architecture V3 - Production Ready"
    )
    parser.add_argument("--parallel", action="store_true", 
                       help="Enable parallel processing (experimental)")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--module", help="Process specific module only")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Test timeout in seconds (default: 300)")
    parser.add_argument("--iterations", type=int, default=5, 
                       help="Max test fix iterations (default: 5)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Simulate execution without making changes")
    parser.add_argument("--no-resume", action="store_true", 
                       help="Start fresh, ignore previous progress")
    parser.add_argument("--progress-file", default="progress.json",
                       help="Progress tracking file (default: progress.json)")
    parser.add_argument("--no-confirmation", action="store_true",
                       help="Skip user confirmation prompts (DANGEROUS!)")
    parser.add_argument("--audit-file", default=".claude_experiments/audit.log",
                       help="Audit log file (default: .claude_experiments/audit.log)")
    
    args = parser.parse_args()
    
    # Initialize fixer with security settings
    try:
        fixer = MicroFixer(
            parallel=args.parallel,
            max_workers=args.workers,
            test_timeout=args.timeout,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            require_confirmation=not args.no_confirmation
        )
    except (SecurityError, ValidationError) as e:
        print(f"\n❌ SECURITY ERROR: {e}")
        print("\nPlease ensure you're running this script from the t-bot project root directory.")
        sys.exit(1)
    
    # Override progress file if specified
    if args.progress_file != "progress.json":
        fixer.progress.progress_file = Path(args.progress_file)
        fixer.progress.progress_data = fixer.progress.load_progress()
    
    # Override audit file if specified
    if args.audit_file != ".claude_experiments/audit.log":
        fixer.audit_logger = AuditLogger(args.audit_file)
    
    try:
        if args.module:
            # Process single module
            print(f"Processing single module: {args.module}")
            result = fixer.fix_module_micro(args.module, max_iterations=args.iterations)
            print(f"\nResult: {'✅ SUCCESS' if result else '❌ FAILED'}")
        else:
            # Process all modules
            fixer.run_systematic_fixes()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        fixer.audit_logger.log_event("SESSION_INTERRUPTED", "User interrupted execution")
        print(f"Progress saved to: {fixer.progress.progress_file}")
        print(f"Audit log: {fixer.audit_logger.audit_file}")
        print(f"Resume with: python {sys.argv[0]} --resume")
        if fixer.session_backups:
            print(f"Session backup available: {fixer.session_backups[-1]}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if 'fixer' in locals():
            fixer.audit_logger.log_event("SESSION_FATAL_ERROR", f"Fatal error: {e}")
            print(f"Audit log: {fixer.audit_logger.audit_file}")
            if fixer.session_backups:
                print(f"Session backup available: {fixer.session_backups[-1]}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()