#!/usr/bin/env python3
"""
PylintErrorParser - Parse and batch pylint errors for processing
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PylintError:
    """Represents a single pylint error"""
    file_path: str
    line_number: int
    column: int
    error_code: str
    error_message: str
    module: str
    
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}:{self.column}: {self.error_code}: {self.error_message}"
    
    @property
    def file_name(self) -> str:
        """Get just the filename without path"""
        return Path(self.file_path).name
    
    @property
    def error_type(self) -> str:
        """Get the general error type (E1101 -> no-member)"""
        error_types = {
            'E1101': 'no-member',
            'E1136': 'unsubscriptable-object',
            'E1120': 'no-value-for-parameter',
            'E1123': 'unexpected-keyword-arg',
            'E0611': 'no-name-in-module',
            'E1130': 'invalid-unary-operand-type',
            'E0601': 'used-before-assignment',
            'E0602': 'undefined-variable'
        }
        return error_types.get(self.error_code, self.error_code)


class PylintErrorParser:
    """Parse and organize pylint error output"""
    
    def __init__(self):
        self.errors: List[PylintError] = []
        
    def parse_log_file(self, log_file: str) -> List[PylintError]:
        """
        Parse a pylint log file and extract errors
        
        Args:
            log_file: Path to the pylint log file
            
        Returns:
            List of PylintError objects
        """
        log_path = Path(log_file)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> List[PylintError]:
        """
        Parse pylint output content and extract errors
        
        Args:
            content: Pylint output as string
            
        Returns:
            List of PylintError objects
        """
        self.errors = []
        current_module = None
        
        for line in content.split('\n'):
            # Check for module line
            module_match = re.match(r'\*+\s*Module\s+(.+)', line)
            if module_match:
                current_module = module_match.group(1)
                continue
            
            # Parse error line
            # Format: src/core/config.py:389:15: E1136: Value 'self.supported_exchanges' is unsubscriptable (unsubscriptable-object)
            error_match = re.match(
                r'([^:]+):(\d+):(\d+):\s+([A-Z]\d+):\s+(.+?)(?:\s+\([^)]+\))?$',
                line
            )
            
            if error_match and current_module:
                file_path = error_match.group(1)
                line_num = int(error_match.group(2))
                column = int(error_match.group(3))
                error_code = error_match.group(4)
                error_msg = error_match.group(5)
                
                error = PylintError(
                    file_path=file_path,
                    line_number=line_num,
                    column=column,
                    error_code=error_code,
                    error_message=error_msg,
                    module=current_module
                )
                self.errors.append(error)
        
        return self.errors
    
    def group_by_file(self) -> Dict[str, List[PylintError]]:
        """
        Group errors by file
        
        Returns:
            Dictionary mapping file paths to their errors
        """
        grouped = {}
        for error in self.errors:
            if error.file_path not in grouped:
                grouped[error.file_path] = []
            grouped[error.file_path].append(error)
        return grouped
    
    def group_by_error_type(self) -> Dict[str, List[PylintError]]:
        """
        Group errors by error type
        
        Returns:
            Dictionary mapping error codes to their instances
        """
        grouped = {}
        for error in self.errors:
            if error.error_code not in grouped:
                grouped[error.error_code] = []
            grouped[error.error_code].append(error)
        return grouped
    
    def batch_errors(self, batch_size: int = 10, strategy: str = 'file') -> List[List[PylintError]]:
        """
        Batch errors for processing
        
        Args:
            batch_size: Maximum number of errors per batch
            strategy: Batching strategy ('file', 'type', 'sequential')
            
        Returns:
            List of error batches
        """
        batches = []
        
        if strategy == 'file':
            # Batch by file - all errors from same file together
            grouped = self.group_by_file()
            for file_path, file_errors in grouped.items():
                # Split large files into multiple batches
                for i in range(0, len(file_errors), batch_size):
                    batch = file_errors[i:i + batch_size]
                    batches.append(batch)
                    
        elif strategy == 'type':
            # Batch by error type - similar errors together
            grouped = self.group_by_error_type()
            for error_type, type_errors in grouped.items():
                for i in range(0, len(type_errors), batch_size):
                    batch = type_errors[i:i + batch_size]
                    batches.append(batch)
                    
        else:  # sequential
            # Simple sequential batching
            for i in range(0, len(self.errors), batch_size):
                batch = self.errors[i:i + batch_size]
                batches.append(batch)
        
        return batches
    
    def get_priority_errors(self) -> List[PylintError]:
        """
        Get high-priority errors that should be fixed first
        
        Returns:
            List of high-priority errors
        """
        # Priority order: undefined variables, import errors, then others
        priority_codes = ['E0602', 'E0601', 'E0611', 'E1101']
        
        priority_errors = []
        for code in priority_codes:
            priority_errors.extend([e for e in self.errors if e.error_code == code])
        
        # Add remaining errors
        remaining = [e for e in self.errors if e.error_code not in priority_codes]
        priority_errors.extend(remaining)
        
        return priority_errors
    
    def filter_errors(self, 
                     error_codes: Optional[List[str]] = None,
                     files: Optional[List[str]] = None) -> List[PylintError]:
        """
        Filter errors by criteria
        
        Args:
            error_codes: List of error codes to include
            files: List of file patterns to include
            
        Returns:
            Filtered list of errors
        """
        filtered = self.errors
        
        if error_codes:
            filtered = [e for e in filtered if e.error_code in error_codes]
        
        if files:
            filtered = [e for e in filtered 
                       if any(pattern in e.file_path for pattern in files)]
        
        return filtered
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of errors
        
        Returns:
            Dictionary with error statistics
        """
        if not self.errors:
            return {
                'total_errors': 0,
                'files_affected': 0,
                'error_types': {},
                'most_common_error': None
            }
        
        error_types = {}
        for error in self.errors:
            error_types[error.error_code] = error_types.get(error.error_code, 0) + 1
        
        most_common = max(error_types.items(), key=lambda x: x[1])
        
        return {
            'total_errors': len(self.errors),
            'files_affected': len(set(e.file_path for e in self.errors)),
            'error_types': error_types,
            'most_common_error': most_common
        }


def main():
    """Test the parser"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pylint_error_parser.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    parser = PylintErrorParser()
    try:
        errors = parser.parse_log_file(log_file)
        
        # Print summary
        summary = parser.get_summary()
        print(f"📊 Error Summary:")
        print(f"  Total errors: {summary['total_errors']}")
        print(f"  Files affected: {summary['files_affected']}")
        if summary['most_common_error']:
            print(f"  Most common: {summary['most_common_error'][0]} ({summary['most_common_error'][1]} occurrences)")
        
        # Show error distribution
        print(f"\n📈 Error Distribution:")
        for code, count in sorted(summary['error_types'].items()):
            print(f"  {code}: {count}")
        
        # Show batching
        batches = parser.batch_errors(batch_size=5, strategy='file')
        print(f"\n📦 Created {len(batches)} batches for processing")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()