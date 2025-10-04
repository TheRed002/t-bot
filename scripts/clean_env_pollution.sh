#!/bin/bash
# Clean environment pollution from placeholder variables

echo "🧹 Cleaning Environment Pollution"
echo "=================================="

# List of placeholder variables detected
PLACEHOLDER_VARS=(
    "OKX_API_KEY"
    "BINANCE_SECRET_KEY"
    "BINANCE_API_KEY"
    "OKX_SECRET_KEY"
    "OKX_PASSPHRASE"
    "COINBASE_API_KEY"
    "COINBASE_SECRET_KEY"
    "COINBASE_PASSPHRASE"
)

echo ""
echo "📋 Variables to clean:"
for var in "${PLACEHOLDER_VARS[@]}"; do
    if [ ! -z "${!var}" ]; then
        echo "  - $var=${!var}"
    fi
done

echo ""
echo "💡 To clean your current shell session, run:"
echo ""
echo "unset ${PLACEHOLDER_VARS[*]}"
echo ""

echo "💡 To prevent this in future sessions:"
echo ""
echo "1. Check your ~/.bashrc or ~/.profile for these exports"
echo "2. Remove any lines that set these variables"
echo "3. Restart your shell"
echo ""

echo "✅ The Config now loads from .env file only (with validation)"
echo "⚠️  Shell must be restarted for changes to take effect"
