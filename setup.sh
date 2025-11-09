#!/bin/bash
# Setup script for ML Zoomcamp Midterm Project
# This script automates the environment setup using uv
#
# Why create a virtual environment with uv?
# - Better VS Code/Jupyter integration (IDE can detect interpreter)
# - Easier kernel registration for notebooks
# - More familiar workflow (activate once, use regular commands)
# - Alternative: use 'uv run' for commands without activation

set -e  # Exit on error

echo "🚀 ML Zoomcamp Midterm Project Setup"
echo "======================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully"
    echo ""
    echo "⚠️  Important: Please restart your terminal or run:"
    echo "   source ~/.bashrc  # or ~/.zshrc"
    echo "   Then run this script again: ./setup.sh"
    exit 0
fi

echo "✅ uv is installed (version: $(uv --version))"
echo ""

# Check if virtual environment already exists
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists (.venv)"
    read -p "   Do you want to recreate it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf .venv
    else
        echo "📦 Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔨 Creating virtual environment with uv..."
    uv venv
    echo "✅ Virtual environment created at .venv/"
else
    echo "✅ Virtual environment exists at .venv/"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📦 Installing project dependencies..."
echo "   This includes: pandas, numpy, scikit-learn, matplotlib, seaborn"
echo "   Plus dev tools: jupyter, ipykernel, pytest"
echo ""
uv sync --all-groups
echo "✅ All dependencies installed"
echo ""

# Register Jupyter kernel for VS Code integration
echo "📓 Registering Jupyter kernel for VS Code/Jupyter..."
python -m ipykernel install --user --name=ml_zoomcamp --display-name "Python (ML Zoomcamp)"
echo "✅ Jupyter kernel 'Python (ML Zoomcamp)' registered"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
echo ""
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo ""
echo "Key packages installed:"
python -c "import pandas; print(f'  ✓ pandas {pandas.__version__}')"
python -c "import numpy; print(f'  ✓ numpy {numpy.__version__}')"
python -c "import sklearn; print(f'  ✓ scikit-learn {sklearn.__version__}')"
python -c "import matplotlib; print(f'  ✓ matplotlib {matplotlib.__version__}')"
python -c "import seaborn; print(f'  ✓ seaborn {seaborn.__version__}')"
python -c "import jupyter; print(f'  ✓ jupyter {jupyter.__version__}')"
echo ""

echo "✅ Setup complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📝 Next Steps:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1️⃣  Activate the virtual environment (do this each time you start):"
echo "   source .venv/bin/activate"
echo ""
echo "2️⃣  Configure VS Code Python interpreter:"
echo "   - Press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows/Linux)"
echo "   - Type 'Python: Select Interpreter'"
echo "   - Choose: ./.venv/bin/python"
echo ""
echo "3️⃣  Open and run the notebook:"
echo "   - Open: notebooks/notebook.ipynb"
echo "   - Select kernel: 'Python (ML Zoomcamp)'"
echo "   - Start running cells!"
echo ""
echo "4️⃣  Or start Jupyter from terminal:"
echo "   jupyter notebook"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "💡 Useful Commands:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Add a new package:"
echo "  uv add package-name"
echo ""
echo "Add development dependency:"
echo "  uv add --dev package-name"
echo ""
echo "Update dependencies:"
echo "  uv sync"
echo ""
echo "Run without activation (alternative):"
echo "  uv run python script.py"
echo "  uv run jupyter notebook"
echo ""
echo "🎉 Happy coding!"
echo ""
