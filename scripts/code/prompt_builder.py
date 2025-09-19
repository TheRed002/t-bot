#!/usr/bin/env python3
"""
PromptBuilder - Build accurate prompts with project context for fixing pylint errors
"""

from pathlib import Path
from typing import List, Optional, Dict
from pylint_error_parser import PylintError


class PromptBuilder:
    """Build context-aware prompts for fixing pylint errors"""
    
    def __init__(self, project_root: Path = None):
        """
        Initialize PromptBuilder
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.reference_cache: Dict[str, str] = {}
        
    def read_reference_md(self, module_name: str) -> Optional[str]:
        """
        Read REFERENCE.md for a module if it exists
        
        Args:
            module_name: Name of the module
            
        Returns:
            Content of REFERENCE.md or None if not found
        """
        if module_name in self.reference_cache:
            return self.reference_cache[module_name]
        
        reference_path = self.project_root / "src" / module_name / "REFERENCE.md"
        
        if reference_path.exists():
            try:
                with open(reference_path, 'r') as f:
                    content = f.read()
                    # Cache for future use
                    self.reference_cache[module_name] = content
                    return content
            except Exception as e:
                print(f"⚠️ Could not read REFERENCE.md for {module_name}: {e}")
        
        return None
    
    def get_module_context(self, module_name: str) -> str:
        """
        Get relevant context for a module
        
        Args:
            module_name: Name of the module
            
        Returns:
            Context string for the module
        """
        reference_content = self.read_reference_md(module_name)
        
        if reference_content:
            # Extract key sections from REFERENCE.md
            context_parts = []
            
            # Extract business context (includes Role, Responsibilities, Critical Notes, Usage Patterns)
            if "## BUSINESS CONTEXT" in reference_content:
                start = reference_content.find("## BUSINESS CONTEXT")
                end = reference_content.find("\n## INTEGRATION", start)
                if end == -1:
                    end = reference_content.find("\n## DETECTED PATTERNS", start)
                if end == -1:
                    end = reference_content.find("\n## MODULE OVERVIEW", start)
                if end == -1:
                    end = len(reference_content)
                business_context = reference_content[start:end].strip()
                if len(business_context) < 3000:  # Increased limit for full context
                    context_parts.append(business_context)
            
            # Extract integration info (Dependencies and Used By)
            if "## INTEGRATION" in reference_content:
                start = reference_content.find("## INTEGRATION")
                end = reference_content.find("\n## ", start + 1)
                if end == -1:
                    end = len(reference_content)
                integration = reference_content[start:end].strip()
                if len(integration) < 1000:
                    context_parts.append(integration)
            
            # Extract detected patterns (Financial, Security, Performance, Architecture)
            if "## DETECTED PATTERNS" in reference_content:
                start = reference_content.find("## DETECTED PATTERNS")
                end = reference_content.find("\n## MODULE OVERVIEW", start)
                if end == -1:
                    end = reference_content.find("\n## COMPLETE API", start)
                if end == -1:
                    end = len(reference_content)
                patterns = reference_content[start:end].strip()
                if len(patterns) < 1000:
                    context_parts.append(patterns)
            
            if context_parts:
                return "\n\n".join(context_parts)
        
        # Fallback to basic module information
        return f"""Module: {module_name}
This is part of a cryptocurrency trading bot system with institutional-grade reliability.
Key principles:
- Use Decimal for all financial calculations, never float
- Maintain strict error handling and validation
- Follow service/repository/controller pattern
- Ensure thread safety for concurrent operations"""
    
    def build_error_fix_prompt(self, 
                              errors: List[PylintError], 
                              module_name: str,
                              batch_number: int = 1,
                              total_batches: int = 1) -> str:
        """
        Build a prompt for fixing a batch of pylint errors
        
        Args:
            errors: List of pylint errors to fix
            module_name: Name of the module being fixed
            batch_number: Current batch number
            total_batches: Total number of batches
            
        Returns:
            Carefully crafted prompt string
        """
        # Get module context
        module_context = self.get_module_context(module_name)
        
        # Group errors by file for better organization
        errors_by_file = {}
        for error in errors:
            if error.file_path not in errors_by_file:
                errors_by_file[error.file_path] = []
            errors_by_file[error.file_path].append(error)
        
        # Format errors for the prompt
        errors_text = ""
        for file_path, file_errors in errors_by_file.items():
            errors_text += f"\n📁 File: {file_path}\n"
            for error in sorted(file_errors, key=lambda e: e.line_number):
                errors_text += f"  Line {error.line_number}: {error.error_code} - {error.error_message}\n"
        
        # Build the prompt
        prompt = f"""Fix the following pylint errors in the {module_name} module (Batch {batch_number}/{total_batches}).

## MODULE CONTEXT
{module_context}

## PROJECT STANDARDS
This is a financial trading system. You MUST:
1. Use Decimal for ALL monetary/price calculations, NEVER float
2. Maintain all existing functionality - do NOT refactor
3. Follow existing patterns in the codebase
4. Preserve all error handling and validation
5. Use proper Python typing and imports
6. Keep changes MINIMAL - only fix the specific errors

## ERRORS TO FIX
{errors_text}

## SPECIFIC FIXES REQUIRED
Based on the error codes, apply these fixes:
"""
        
        # Add specific guidance for each error type
        error_codes = set(e.error_code for e in errors)
        
        if 'E0611' in error_codes:  # no-name-in-module
            prompt += """
- E0611 (no-name-in-module): Check if imports are correct. The module structure may have changed.
  Look for circular imports or incorrect relative imports."""
        
        if 'E1101' in error_codes:  # no-member
            prompt += """
- E1101 (no-member): The attribute/method doesn't exist. Either:
  - Add the missing attribute/method
  - Fix the attribute name (typo)
  - Add proper type hints to help pylint understand"""
        
        if 'E0601' in error_codes or 'E0602' in error_codes:  # undefined
            prompt += """
- E0601/E0602 (undefined variable): Variable used before assignment. Either:
  - Initialize the variable before use
  - Import missing names
  - Fix variable scope issues"""
        
        if 'E1136' in error_codes:  # unsubscriptable-object
            prompt += """
- E1136 (unsubscriptable): Object cannot be indexed/sliced. Either:
  - Ensure the object is the correct type (list/dict/tuple)
  - Add type hints to clarify the object type
  - Check if the object needs to be converted first"""
        
        if 'E1120' in error_codes or 'E1123' in error_codes:  # argument errors
            prompt += """
- E1120/E1123 (argument errors): Function call has wrong arguments. Either:
  - Fix the function call to match the signature
  - Update the function signature if needed
  - Remove/add required arguments"""
        
        if 'E1130' in error_codes:  # invalid-unary-operand
            prompt += """
- E1130 (invalid unary operand): The operation is invalid for this type. Either:
  - Check for None values before operations
  - Ensure proper type conversion
  - Add validation before the operation"""
        
        prompt += """

## IMPORTANT REMINDERS
1. DO NOT create new files or move code between files
2. DO NOT refactor or "improve" code beyond fixing the errors
3. PRESERVE all existing logic and behavior
4. Make the MINIMUM changes necessary
5. If an import is missing, add it at the top of the file
6. If a type hint would help pylint understand, add it
7. This is a PRODUCTION financial system - accuracy is critical

Fix these specific errors only. Do not make any other changes."""
        
        return prompt
    
    def build_verification_prompt(self, module_name: str) -> str:
        """
        Build a prompt to verify all fixes are complete
        
        Args:
            module_name: Name of the module
            
        Returns:
            Verification prompt string
        """
        return f"""Verify that all pylint errors have been fixed in the {module_name} module.

Run a final check to ensure:
1. All imports are correct
2. All variables are defined before use
3. All function calls have correct arguments
4. All type operations are valid
5. No new errors were introduced

If any errors remain, fix them following the same principles:
- Minimal changes only
- Preserve all functionality
- Use Decimal for financial calculations
- Follow existing patterns"""


def main():
    """Test the prompt builder"""
    import sys
    from pylint_error_parser import PylintErrorParser
    
    if len(sys.argv) < 2:
        print("Usage: python prompt_builder.py <module_name> [log_file]")
        sys.exit(1)
    
    module_name = sys.argv[1]
    log_file = sys.argv[2] if len(sys.argv) > 2 else f"src/{module_name}/{module_name}_pylint.logs"
    
    # Parse errors
    parser = PylintErrorParser()
    try:
        errors = parser.parse_log_file(log_file)
        
        if not errors:
            print("No errors found")
            return
        
        # Build prompt
        builder = PromptBuilder()
        
        # Create batches and show first prompt
        batches = parser.batch_errors(batch_size=5, strategy='file')
        
        if batches:
            prompt = builder.build_error_fix_prompt(
                batches[0], 
                module_name,
                batch_number=1,
                total_batches=len(batches)
            )
            
            print("📝 Generated Prompt:")
            print("-" * 80)
            print(prompt)
            print("-" * 80)
            print(f"\n📊 Prompt length: {len(prompt)} characters")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()