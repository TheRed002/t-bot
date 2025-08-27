#!/bin/bash

# Script to disable GitHub CI/CD workflows (for Bitbucket migration)

echo "Disabling GitHub CI/CD workflows..."

# Create disabled directory if it doesn't exist
if [ -d ".github/workflows" ]; then
    mkdir -p .github/workflows.disabled
    
    # Move all workflow files
    if ls .github/workflows/*.yml 1> /dev/null 2>&1; then
        mv .github/workflows/*.yml .github/workflows.disabled/
        echo "✅ Moved all workflow files to .github/workflows.disabled/"
    else
        echo "⚠️ No workflow files found to disable"
    fi
    
    # Create a README to explain the change
    cat > .github/workflows/README.md << 'EOF'
# GitHub Workflows Disabled

The GitHub CI/CD workflows have been disabled as this project uses Bitbucket for version control.

All workflow files have been moved to `.github/workflows.disabled/` for reference.

To re-enable GitHub workflows:
1. Move the desired workflow files from `workflows.disabled/` back to `workflows/`
2. Delete this README file
3. Ensure GitHub Actions is enabled in the repository settings
EOF
    
    echo "✅ Created README in workflows directory"
else
    echo "⚠️ No .github/workflows directory found"
fi

echo "✅ GitHub CI/CD workflows disabled successfully"