# Pylint Fixer - Production Ready

## Overview
A robust, iterative pylint error fixer that uses Claude to automatically fix Python code issues in the T-Bot trading system.

## Features

### ✅ Safety Features
- **Path traversal protection**: Blocks dangerous paths like `../`, `/etc/`, etc.
- **Timeout protection**: 15-minute default timeout for Claude calls
- **Prompt size limits**: Maximum 50KB prompts to prevent overload
- **Progress tracking**: Detects when no progress is made and stops
- **Dry-run mode**: Test without making actual changes
- **File backup**: Original files preserved (pylint doesn't modify)
- **Confirmation prompts**: Asks before modifying files

### ✅ Robustness
- **Handles subdirectories**: Recursively scans all Python files
- **Batch processing**: Processes errors in configurable batches
- **Retry logic**: Continues even if some batches fail
- **Memory efficient**: Cleans up resources properly
- **Unicode support**: Handles international characters
- **Concurrent safe**: Multiple instances don't interfere

### ✅ Error Handling
- **Empty modules**: Handles gracefully
- **Large error counts**: Batches automatically
- **Permission errors**: Reports but continues
- **Missing modules**: Lists available modules
- **Network failures**: Reports Claude connection issues

## Installation

1. Ensure Claude CLI is installed and in PATH
2. Install Python dependencies:
```bash
pip install pylint
```

## Usage

### Basic Check (No Changes)
```bash
# From project root
python scripts/code/pylint_fixer_final.py core

# Check specific module
python scripts/code/pylint_fixer_final.py utils
```

### Fix Errors
```bash
# Fix with confirmation prompt
python scripts/code/pylint_fixer_final.py core --fix

# Dry run (simulate without changes)
python scripts/code/pylint_fixer_final.py core --fix --dry-run

# Custom settings
python scripts/code/pylint_fixer_final.py core --fix \
  --max-iterations 5 \
  --batch-size 5 \
  --timeout 600
```

### Options
- `module`: Module name to check/fix (required)
- `--fix`: Actually fix errors (without this, just reports)
- `--max-iterations N`: Maximum fix iterations (default: 10)
- `--batch-size N`: Errors per batch (default: 10)
- `--batch-strategy`: How to batch: 'file', 'type', 'sequential' (default: file)
- `--model`: Claude model to use (default: claude-sonnet-4-20250514)
- `--timeout N`: Claude timeout in seconds (default: 900)
- `--dry-run`: Simulate without calling Claude
- `--no-recursive`: Don't scan subdirectories
- `--project-root`: Explicitly set project root

## Batching Strategies

- **file**: Groups errors by file (recommended)
- **type**: Groups similar error types together
- **sequential**: Processes errors in order found

## Error Types Handled

- **E0601/E0602**: Undefined variables
- **E0611**: Import errors
- **E1101**: No member/attribute errors
- **E1120/E1123**: Function argument errors
- **E1128**: Assignment from None
- **E1130**: Invalid operations
- **E1136**: Unsubscriptable objects

## File Structure

```
scripts/code/
├── pylint_fixer_final.py        # Main production script
├── pylint_fixer_orchestrator_v2.py  # Orchestration logic
├── claude_executor_v2.py        # Claude API interface
├── pylint_error_parser.py       # Error parsing
├── prompt_builder.py            # Prompt generation
├── test_pylint_fixer.py        # Unit tests
└── test_edge_cases.py          # Edge case tests
```

## Safety Considerations

1. **Always review changes**: While Claude is good, verify the fixes
2. **Use version control**: Commit before running fixes
3. **Start small**: Test on one module before running on all
4. **Monitor iterations**: If it takes many iterations, manual review may be needed
5. **Check test suite**: Run tests after fixes to ensure functionality

## Process Flow

1. **Scan**: Runs pylint to find all errors
2. **Parse**: Extracts and categorizes errors
3. **Batch**: Groups errors for efficient processing
4. **Context**: Reads REFERENCE.md for module understanding
5. **Prompt**: Generates targeted fix prompts
6. **Execute**: Sends to Claude for fixing
7. **Iterate**: Repeats until clean or max iterations
8. **Report**: Provides detailed statistics

## Typical Output

```
📁 Project root: /path/to/project
================================================================================
🔍 Checking module: core
================================================================================
📂 Found 63 Python files to scan
🔍 Running pylint on /path/to/project/src/core...

📊 Error Summary:
  Total errors: 64
  Files affected: 21
  Most common: E1101 (41 occurrences)

📈 Error Distribution:
  E0601: 6
  E0602: 1
  ...

🚀 Starting fix process...
📍 Iteration 1/10
----------------------------------------
📊 Found 64 errors in 21 files
📦 Created 8 batch(es) for processing
  🔧 Processing batch 1/8 (10 errors)
  ✅ Batch 1 processed
  ...

✅ All errors fixed! Module core is clean.
```

## Testing

Run comprehensive tests:
```bash
# Unit tests
python scripts/code/test_pylint_fixer.py

# Edge case tests  
python scripts/code/test_edge_cases.py
```

## Troubleshooting

### "Claude binary not found"
- Ensure `claude` is installed and in PATH
- Test with: `claude --version`

### "No progress for 3 iterations"
- Some errors may require manual fixing
- Try different batch strategy
- Reduce batch size

### "Module not found"
- Check available modules listed
- Ensure running from project root
- Use --project-root if needed

### "Timeout" errors
- Increase timeout with --timeout
- Reduce batch size
- Check network connection

## Best Practices

1. **Fix modules incrementally**: Don't try to fix all modules at once
2. **Review critical modules**: Manually review fixes for core, risk_management, execution
3. **Use dry-run first**: Test with --dry-run before actual fixes
4. **Keep logs**: Save output for review
5. **Update tests**: Ensure tests still pass after fixes

## Module Priority

Recommended order for fixing:
1. `utils` - Utilities (safest to start)
2. `error_handling` - Error handling
3. `database` - Database layer
4. `monitoring`, `state` - Infrastructure
5. `core` - Core functionality
6. `data` - Data processing
7. `exchanges` - Exchange connections
8. `risk_management` - Risk controls (careful review needed)
9. `execution` - Order execution (careful review needed)
10. `strategies` - Trading strategies
11. `web_interface` - Web interface

## Notes

- Financial calculations must use Decimal, never float
- All changes preserve existing functionality
- Minimal changes policy - only fixes specific errors
- No refactoring or "improvements" beyond error fixes
- Thread safety maintained for concurrent operations

## Support

For issues or questions:
1. Check error logs in `src/MODULE/MODULE_pylint.logs`
2. Review fix summary in `src/MODULE/MODULE_fix_summary.txt`
3. Run with --dry-run to diagnose without changes
4. Use smaller batch sizes for problematic modules