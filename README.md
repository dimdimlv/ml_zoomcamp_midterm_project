# Predicting Online Purchase Intention (ML Zoomcamp Midterm Project)

## 1. Problem Description
This project predicts whether a website visitor will make a purchase during an online shopping session.

It is a **binary classification problem** with target variable `Revenue` (1 = purchase, 0 = no purchase). The goal is to help e-commerce businesses identify high-intent users to optimize remarketing and ad targeting.

### Why It Matters
- Enables smarter bidding and conversion optimization.
- Improves customer segmentation and personalization.
- Reduces marketing costs by focusing on high-probability buyers.

---

## 2. Dataset
**Source:** [Online Shoppers Purchasing Intention Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

**Description:**
- Contains 12,330 session records from an e-commerce website.
- Each row represents a session with behavioral, temporal, and technical attributes.
- The target `Revenue` indicates whether a purchase occurred.

**Key Features:**
| Feature | Type | Description |
|----------|------|-------------|
| `Administrative`, `Informational`, `ProductRelated` | Numerical | Number of pages visited of each type |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | Numerical | Time spent on each type of page |
| `BounceRates`, `ExitRates` | Numerical | Average exit/bounce rate for the pages |
| `PageValues` | Numerical | Value of a page (proxy for conversion funnel position) |
| `SpecialDay` | Numerical | Proximity to special days (e.g., Valentine’s Day) |
| `Month` | Categorical | Month of the visit |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Categorical | Session context features |
| `VisitorType` | Categorical | Returning or new visitor |
| `Weekend` | Boolean | Whether the session occurred on a weekend |
| `Revenue` | Boolean | **Target variable** — purchase made (1) or not (0) |

**Size:** 12,330 rows × 18 columns

---

## 3. Approach & Methods
### Workflow
1. **EDA & Preprocessing**
   - Check distributions, correlations, missing values.
   - Convert categorical variables.
   - Scale numeric features.

2. **Modeling**
   - Baseline: Logistic Regression with class balancing.
   - Compare tree-based models (RandomForest, XGBoost).
   - Metrics: ROC-AUC, F1, Precision, Recall.

3. **Deployment**
   - Refactored into modular scripts:
     - `train.py` — model training and saving.
     - `predict.py` — single prediction from JSON input.
     - `serve.py` — FastAPI service exposing `/predict` and `/health`.
   - Containerized with Docker.

---

## 4. Folder Structure
```
.
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ data/
│  └─ online_shoppers_intention.csv
├─ src/
│  ├─ train.py
│  ├─ predict.py
│  └─ serve.py
├─ models/
│  └─ model.joblib
└─ tests/
   └─ test_api.http
```

---

## 5. Installation & Usage

### Prerequisites
- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Quick Start (Automated Setup)

The easiest way to set up the project is using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/dimdimlv/ml_zoomcamp_midterm_project.git
cd ml_zoomcamp_midterm_project

# Run the automated setup script
chmod +x setup.sh
./setup.sh
```

This script will:
1. Install `uv` (if not already installed)
2. Create a virtual environment at `.venv/`
3. Install all dependencies (base + dev + web + ml)
4. Register the Jupyter kernel for VS Code
5. Verify the installation

**After setup completes:**
```bash
# Activate the environment
source .venv/bin/activate

# Start working!
jupyter notebook  # Or open notebooks/notebook.ipynb in VS Code
```

---

### Manual Setup

If you prefer to set up manually:

### Install uv (if not already installed)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew
brew install uv

# Or using pip
pip install uv
```

### Project Setup with uv

#### 1. Clone the Repository
```bash
git clone https://github.com/dimdimlv/ml_zoomcamp_midterm_project.git
cd ml_zoomcamp_midterm_project
```

#### 2. Create Virtual Environment
```bash
# Create a virtual environment with uv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

**Why create a virtual environment with uv?**
- ✅ Better VS Code/Jupyter integration (IDE can detect the interpreter)
- ✅ Easier Jupyter kernel registration for notebooks
- ✅ Familiar workflow: activate once, use regular Python commands
- 💡 Alternative: Use `uv run` for commands without activation (e.g., `uv run python script.py`)

#### 3. Install Dependencies

**Option A: Using pyproject.toml (Recommended)**
```bash
# Install all project dependencies
uv sync

# Or install with specific dependency groups
uv sync --group dev          # Add development dependencies (Jupyter, pytest)
uv sync --group web          # Add web service dependencies (FastAPI, uvicorn)
uv sync --group ml           # Add advanced ML dependencies (XGBoost, SHAP)
uv sync --all-groups         # Install all dependencies
```

**Option B: Install packages manually**
```bash
# Add production dependencies
uv add pandas numpy matplotlib seaborn scikit-learn

# Add development dependencies (Jupyter, testing, etc.)
uv add --dev jupyter ipykernel pytest black isort

# Add web service dependencies (if deploying)
uv add fastapi uvicorn[standard] pydantic

# Add advanced ML libraries
uv add xgboost lightgbm shap
```

**Understanding Dependency Groups:**
- **Base dependencies**: Core ML libraries (pandas, numpy, scikit-learn, etc.)
- **dev**: Development tools (Jupyter notebooks, testing frameworks)
- **web**: Web deployment (FastAPI service)
- **ml**: Advanced ML models and interpretability tools

#### 4. Configure Jupyter Kernel (for notebook development)
```bash
# Register the kernel with Jupyter
python -m ipykernel install --user --name=ml_zoomcamp --display-name "Python (ML Zoomcamp)"
```

#### 5. Set Python Interpreter in VS Code
1. Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv` (path: `./.venv/bin/python`)

### Running the Project

#### Train the Model
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run training script
python src/train.py --data-path data/online_shoppers_intention.csv
```

#### Run Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Or use VS Code's built-in Jupyter support
# Open notebooks/notebook.ipynb and select the "Python (ML Zoomcamp)" kernel
```

#### Start the API Service
```bash
# Using uvicorn
uvicorn src.serve:app --host 0.0.0.0 --port 9696 --reload
```

### Quick Reference Commands

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Add a new package
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update all dependencies
uv lock --upgrade

# Show installed packages
uv pip list

# Remove a package
uv remove package-name

# Run Python script with uv
uv run python src/train.py
```

### Alternative: Traditional pip Setup
If you prefer using pip instead of uv:
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,web,ml]"  # Install all optional dependencies
# Or
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Example API Request
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 3,
    "Informational": 0,
    "ProductRelated": 10,
    "Administrative_Duration": 45.0,
    "Informational_Duration": 0.0,
    "ProductRelated_Duration": 210.0,
    "BounceRates": 0.02,
    "ExitRates": 0.04,
    "PageValues": 25.6,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

**Response:**
```json
{"purchase_probability": 0.72, "will_purchase": true}
```

### Docker Setup
```bash
docker build -t shoppers-intent .
docker run --rm -p 9696:9696 shoppers-intent
```

---

## 6. Model Evaluation (example metrics)
| Metric | Score |
|---------|--------|
| ROC-AUC | 0.89 |
| F1 Score | 0.73 |
| Precision | 0.75 |
| Recall | 0.70 |

---

## 7. Frequently Asked Questions (FAQ)

### Q: Why use `uv` instead of `pip`?
**A:** `uv` is significantly faster (10-100x) than pip, has better dependency resolution, and provides modern project management features. However, you can still use pip if preferred - see the "Alternative: Traditional pip Setup" section.

### Q: Do I need to create a virtual environment with uv?
**A:** Not strictly required (uv can auto-manage environments with `uv run`), but **recommended for this project** because:
- Better integration with VS Code and Jupyter notebooks
- Easier kernel registration
- More familiar workflow for most developers

### Q: What's the difference between `uv add` and `uv pip install`?
**A:** 
- `uv add`: Adds packages to `pyproject.toml` and `uv.lock` (recommended for projects)
- `uv pip install`: Just installs packages (pip-compatible, doesn't update project files)
- Use `uv add` for permanent dependencies, `uv pip install` for quick tests

### Q: How do I update my dependencies?
**A:**
```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Then sync your environment
uv sync
```

### Q: The setup script fails. What should I do?
**A:** 
1. Make sure you have Python 3.10+ installed: `python --version`
2. Install uv manually: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Restart your terminal
4. Try running the setup script again: `./setup.sh`
5. Or follow the manual setup instructions in the README

### Q: How do I run the notebook in VS Code?
**A:**
1. Install Python and Jupyter extensions in VS Code
2. Open `notebooks/notebook.ipynb`
3. Click the kernel selector (top right)
4. Choose "Python (ML Zoomcamp)" or select `.venv/bin/python`

---

## 8. Next Steps / Future Work
- Try **XGBoost** or **CatBoost** models.
- Use **SMOTE** or class weighting for imbalance.
- Add **SHAP** or **LIME** interpretability.
- Deploy to **Render / Railway**.
- Add automated testing (pytest + REST client).

---

## 9. References
- UCI ML Repository: Online Shoppers Intention Dataset
- DataTalksClub ML Zoomcamp guidelines

### Dataset citation (BibTeX)
If you use this dataset in a paper or project, please cite it as follows:

```bibtex
@misc{online_shoppers_purchasing_intention_dataset_468,
   author       = {Sakar, C. and Kastro, Yomi},
   title        = {{Online Shoppers Purchasing Intention Dataset}},
   year         = {2018},
   howpublished = {UCI Machine Learning Repository},
   note         = {{DOI}: https://doi.org/10.24432/C5F88Q}
}
```

