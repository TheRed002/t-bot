#!/usr/bin/env python3
"""Update config.schema.json to remove hardcoded credential defaults."""

import json
from pathlib import Path

def update_schema_defaults():
    """Remove hardcoded credential defaults from schema."""
    schema_path = Path("config/config.schema.json")

    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return False

    # Load schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Credentials to clear
    credentials = [
        'binance_api_key',
        'binance_api_secret',
        'coinbase_api_key',
        'coinbase_api_secret',
        'okx_api_key',
        'okx_api_secret',
        'okx_passphrase',
    ]

    # Track changes
    changes = []

    # Update exchanges section
    if 'properties' in schema and 'exchanges' in schema['properties']:
        exchanges = schema['properties']['exchanges']
        if 'default' in exchanges:
            for cred in credentials:
                if cred in exchanges['default']:
                    old_value = exchanges['default'][cred]
                    if old_value and 'your_' in old_value:
                        exchanges['default'][cred] = ""
                        changes.append(f"  - {cred}: '{old_value}' → ''")

    # Update $defs/ExchangeConfig if it exists
    if '$defs' in schema and 'ExchangeConfig' in schema['$defs']:
        exchange_config = schema['$defs']['ExchangeConfig']
        if 'properties' in exchange_config:
            for cred in credentials:
                if cred in exchange_config['properties']:
                    prop = exchange_config['properties'][cred]
                    if 'default' in prop and prop['default'] and 'your_' in prop['default']:
                        old_value = prop['default']
                        prop['default'] = ""
                        changes.append(f"  - {cred}: '{old_value}' → ''")

    if not changes:
        print("✅ No hardcoded credentials found in schema")
        return True

    # Save updated schema
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"✅ Updated {len(changes)} credential defaults in schema:")
    for change in changes:
        print(change)

    return True

if __name__ == "__main__":
    import sys
    success = update_schema_defaults()
    sys.exit(0 if success else 1)
