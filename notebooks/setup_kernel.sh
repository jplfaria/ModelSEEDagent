#!/bin/bash

# Setup script for ModelSEEDagent Jupyter kernel
# Run this from the ModelSEEDagent project root directory

echo "🔧 Setting up ModelSEEDagent Jupyter kernel..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Creating one..."
    python -m venv venv
    source venv/bin/activate
    echo "✅ Created and activated virtual environment"
fi

# Install required packages
echo "📦 Installing ModelSEEDagent and dependencies..."
pip install -e .[all]

# Install ipykernel if not already installed
echo "🔧 Installing ipykernel..."
pip install ipykernel

# Add virtual environment as Jupyter kernel
echo "🎯 Adding ModelSEEDagent kernel to Jupyter..."
python -m ipykernel install --user --name=modelseed-agent --display-name="ModelSEEDagent"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Launch Jupyter: jupyter notebook"
echo "2. Open notebooks/comprehensive_tutorial.ipynb"
echo "3. Select 'ModelSEEDagent' kernel from Kernel menu"
echo "4. Run the tutorial cells"
echo ""
echo "💡 Alternative: If you want to use your current Python environment,"
echo "   just run: pip install -e .[all] from the project root"
