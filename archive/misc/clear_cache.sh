#!/bin/bash
# Clear all caches between testing iterations

echo "🧹 Clearing ModelSEEDagent caches..."

# Clear Python bytecode caches
echo "  - Clearing Python bytecode caches..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clear pytest cache
echo "  - Clearing pytest cache..."
rm -rf .pytest_cache

# Clear CLI configuration and sessions
echo "  - Clearing CLI state..."
rm -f ~/.modelseed-agent-cli.json
rm -rf sessions/ 2>/dev/null || true
rm -rf logs/ 2>/dev/null || true

echo "✅ Cache clearing complete!"
echo "💡 Now start a fresh terminal and test the CLI"
