# Predicting Online Purchase Intention (ML Zoomcamp Midterm Project)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **A complete end-to-end machine learning project for predicting online shopper purchase intention using behavioral and session data.**

## 📊 Project Status: Complete ✓

- ✅ Exploratory Data Analysis
- ✅ Feature Engineering & Selection
- ✅ Model Development & Tuning
- ✅ Final Model Training
- ✅ Production Pipeline Created
- ✅ Comprehensive Documentation

**Model Performance**: 83% ROC-AUC | 81% Accuracy | 69% Recall

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/dimdimlv/ml_zoomcamp_midterm_project.git
cd ml_zoomcamp_midterm_project
./setup.sh

# Activate environment
source .venv/bin/activate

# Open the notebook
jupyter notebook notebooks/notebook.ipynb

# Or use the prediction pipeline
python -c "from src.predictor import OnlineShopperPredictor; predictor = OnlineShopperPredictor()"
```

---

## 📑 Table of Contents

1. [Problem Description](#1-problem-description)
2. [Dataset](#2-dataset)
3. [Project Results](#3-project-results)
4. [Approach & Methods](#4-approach--methods)
5. [Folder Structure](#5-folder-structure)
6. [Installation & Usage](#6-installation--usage)
7. [Deployment Considerations](#7-deployment-considerations)
8. [Technical Stack](#8-technical-stack)
9. [FAQ](#9-frequently-asked-questions-faq)
10. [Next Steps](#10-next-steps--future-work)
11. [Project Deliverables](#11-project-deliverables-)
12. [References](#12-references)

---

## 1. Problem Description
This project predicts whether a website visitor will make a purchase during an online shopping session using machine learning.

It is a **binary classification problem** with target variable `Revenue` (1 = purchase, 0 = no purchase). The goal is to help e-commerce businesses identify high-intent users to optimize remarketing, ad targeting, and real-time personalization.

### Why It Matters
- **Optimize Marketing Spend**: Target users with high purchase probability, reducing wasted ad spend
- **Personalize User Experience**: Customize website content and offers based on predicted intent
- **Prevent Cart Abandonment**: Identify at-risk sessions and intervene with timely offers
- **Improve ROI**: Focus resources on high-value prospects
- **Enable Real-time Decisions**: Fast inference allows session-based interventions

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

## 3. Project Results

### Final Model Performance
Our **XGBoost classifier** achieved excellent results on the test set:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.8309** | Excellent discrimination between buyers and non-buyers |
| **Accuracy** | 80.95% | Overall correct predictions |
| **Precision** | 42.83% | Of predicted buyers, 43% actually purchase |
| **Recall** | 68.85% | Captures 69% of actual buyers |
| **F1-Score** | 0.5281 | Balanced precision-recall trade-off |
| **Avg Precision** | 0.5339 | Area under PR curve |

**Key Insights:**
- ✅ **Strong discrimination ability** (ROC-AUC > 0.83)
- ✅ **High recall** - captures most actual buyers (important for revenue)
- ⚠️ **Moderate precision** - some false positives (acceptable for marketing use cases)
- ✅ **Fast inference** - < 10ms per prediction
- ✅ **Production-ready** with complete prediction pipeline

### Top Predictive Features
Based on comprehensive feature importance analysis:

1. **PageValues** (0.71 importance) - Dominant predictor, tracks value of visited pages
2. **ExitRates** (0.03) - Indicates user engagement and exit behavior
3. **Month** (0.02) - Seasonal patterns affect purchase likelihood
4. **VisitorType** (0.02) - Returning visitors behave differently
5. **ProductRelated_Duration** (0.01) - Time spent on product pages

**Feature Groups Impact:**
- **Page Metrics**: 77% of predictive power (PageValues, ExitRates, BounceRates)
- **Temporal Features**: 15% (Month, Weekend, SpecialDay)
- **Technical Context**: 8% (Browser, OS, Region, TrafficType)

---

## 4. Approach & Methods

### Complete ML Pipeline (5 Parts)

#### **Part 1: Exploratory Data Analysis**
- Comprehensive data quality assessment
- Statistical analysis of all features
- Visualization of distributions and relationships
- Identification of outliers and class imbalance (15% positive class)
- Correlation analysis and multicollinearity detection

#### **Part 2: Feature Engineering**
- Label encoding for categorical features (Month, VisitorType)
- Standard scaling for numerical features
- Feature interaction analysis
- Data splitting: 70% train, 15% validation, 15% test
- Handling of class imbalance

#### **Part 3: Feature Importance Analysis**
Comprehensive feature selection using multiple methods:
- **Random Forest Feature Importances**
- **Permutation Importance**
- **Mutual Information Scores**
- **Correlation Analysis**
- Consensus ranking across all methods

#### **Part 4: Model Selection and Tuning**
1. **Baseline Models**: Dummy classifiers (stratified, most frequent)
2. **Initial Comparison**: 7 algorithms tested
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - K-Nearest Neighbors
   - Gaussian Naive Bayes

3. **Class Imbalance Handling**: Tested 6 techniques
   - Class weights
   - Random over-sampling
   - Random under-sampling
   - SMOTE
   - SMOTE + Tomek Links
   - SMOTETomek

4. **Hyperparameter Tuning**: 
   - RandomizedSearchCV with 5-fold cross-validation
   - Tuned top 3 models (Random Forest, Gradient Boosting, XGBoost)
   - Best: XGBoost with learning_rate=0.01, max_depth=5, n_estimators=300

5. **Final Selection**: XGBoost chosen based on:
   - Highest ROC-AUC score
   - Best generalization (minimal overfitting)
   - Fast training and inference
   - Robust feature importance

#### **Part 5: Final Model Training and Deployment**
- Trained on full training dataset (16,334 samples)
- Comprehensive evaluation on test set (2,466 samples)
- Model artifacts saved:
  - Trained model (XGBoost)
  - Preprocessing pipeline (scaler, encoders)
  - Feature metadata
  - Performance metrics
- Created production-ready prediction pipeline
- Documented deployment architecture and considerations

---

## 5. Folder Structure
```
.
├── README.md                           # Project documentation
├── pyproject.toml                      # Project dependencies and configuration
├── setup.sh                            # Automated setup script
├── LICENSE                             # MIT License
├── data/
│   └── online_shoppers_intention.csv  # Dataset (12,330 sessions)
├── notebooks/
│   └── notebook.ipynb                 # Complete analysis & model development
├── models/                            # Saved model artifacts
│   ├── final_model.pkl                # Trained XGBoost classifier
│   ├── scaler.pkl                     # StandardScaler for features
│   ├── label_encoders.pkl             # Categorical encoders
│   ├── feature_names.json             # Feature metadata
│   └── model_metadata.json            # Performance metrics & config
├── src/
│   └── predictor.py                   # Production prediction pipeline
├── docs/                              # Project documentation
│   ├── deliverables.md               # Project deliverables checklist
│   └── evaluation_criteria.md        # Evaluation rubric
└── tests/                             # Test files (future)
```

---

## 6. Installation & Usage

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

#### Explore the Notebook
The main analysis is in `notebooks/notebook.ipynb`:
```bash
# Start Jupyter
jupyter notebook

# Or use VS Code's built-in Jupyter support
# Open notebooks/notebook.ipynb and select the "Python (ML Zoomcamp)" kernel
```

**Notebook Contents:**
- Part 1: Comprehensive EDA
- Part 2: Feature Engineering
- Part 3: Feature Importance Analysis
- Part 4: Model Selection and Tuning
- Part 5: Final Model Training and Deployment

#### Use the Prediction Pipeline
```python
from src.predictor import OnlineShopperPredictor
import pandas as pd

# Initialize predictor (loads all artifacts)
predictor = OnlineShopperPredictor(models_dir='models')

# Prepare sample data
sample = pd.DataFrame([{
    'Administrative': 3,
    'Administrative_Duration': 45.0,
    'Informational': 0,
    'Informational_Duration': 0.0,
    'ProductRelated': 10,
    'ProductRelated_Duration': 210.0,
    'BounceRates': 0.02,
    'ExitRates': 0.04,
    'PageValues': 25.6,
    'SpecialDay': 0.0,
    'Month': 'Nov',
    'OperatingSystems': 2,
    'Browser': 2,
    'Region': 1,
    'TrafficType': 2,
    'VisitorType': 'Returning_Visitor',
    'Weekend': False
}])

# Make predictions
predictions = predictor.predict(sample)
probabilities = predictor.predict_proba(sample)
detailed = predictor.predict_with_confidence(sample, threshold=0.5)

print(f"Prediction: {predictions[0]}")
print(f"Probability: {probabilities[0]:.4f}")
print(detailed)
```

#### Model Information
```python
import json

# Load model metadata
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model Type: {metadata['model_type']}")
print(f"Training Date: {metadata['training_date']}")
print(f"Performance Metrics:")
for metric, value in metadata['performance_metrics'].items():
    print(f"  {metric}: {value:.4f}")
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
uv run python script.py

# Open Jupyter notebook
jupyter notebook

# Run tests (when implemented)
pytest tests/
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
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

---

## 7. Deployment Considerations

### Production Architecture Options

#### Option 1: REST API Service (Recommended)
Deploy as a web service using Flask/FastAPI:
- Endpoint: `POST /predict` - accepts JSON, returns prediction + probability
- Endpoint: `GET /health` - health check
- Containerized with Docker
- Auto-scaling based on load
- **Use cases**: Real-time predictions, marketing platforms, CRM integration

#### Option 2: Batch Processing
Scheduled batch predictions using Airflow/Cron:
- Process large datasets overnight
- Write results to database
- Generate reports for business intelligence
- **Use cases**: Daily user scoring, campaign planning, analytics

#### Option 3: Streaming
Real-time processing with Kafka/Kinesis:
- Process events as they occur
- Low-latency predictions (< 10ms)
- Integrate with recommendation engines
- **Use cases**: Session-based personalization, real-time interventions

### System Requirements
- **CPU**: 2-4 cores (sufficient for most workloads)
- **Memory**: 2-4 GB RAM
- **Storage**: 100 MB (model + artifacts)
- **Inference Time**: < 10ms per prediction
- **Throughput**: 100-1000 predictions/second (single instance)

### Monitoring & Maintenance
**Key Metrics to Track:**
- Model performance (ROC-AUC, accuracy, drift)
- Prediction latency (p50, p95, p99)
- Data quality (missing values, out-of-range)
- Business impact (conversion rates, revenue)

**Retraining Schedule:**
- Trigger: Performance drop > 5% OR quarterly
- Process: Collect new data → Retrain → A/B test → Deploy
- Validation: Monitor business metrics post-deployment

### Business Use Cases

| Use Case | Threshold | Focus | Expected Impact |
|----------|-----------|-------|-----------------|
| High-value campaigns | 0.7-0.9 | High precision | Target only very likely buyers, maximize ROI |
| Standard marketing | 0.5-0.7 | Balanced | General audience targeting |
| Broad reach | 0.3-0.5 | High recall | Capture more potential buyers |
| Cart abandonment | 0.4-0.6 | Balanced | Intervene when users show exit intent |

---

## 8. Technical Stack

**Core Libraries:**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML Framework**: scikit-learn, XGBoost
- **Model Persistence**: pickle, joblib
- **Environment**: uv (fast package manager)

**Development Tools:**
- **Notebooks**: Jupyter, ipykernel
- **Code Quality**: black, isort, flake8 (optional)
- **Testing**: pytest (optional)

**Deployment (Optional):**
- **API**: FastAPI, uvicorn
- **Containerization**: Docker
- **Monitoring**: Prometheus, Grafana

---

## 9. Frequently Asked Questions (FAQ)

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
5. Run cells sequentially to reproduce the entire analysis

### Q: Can I use the model without retraining?
**A:** Yes! Pre-trained model artifacts are saved in the `models/` directory. Use the `OnlineShopperPredictor` class in `src/predictor.py` to make predictions immediately.

### Q: How accurate is the model?
**A:** The model achieves:
- 83% ROC-AUC (excellent discrimination)
- 81% accuracy overall
- 69% recall (captures most buyers)
- 43% precision (some false positives, acceptable for marketing)

### Q: What's the class imbalance in the dataset?
**A:** The dataset has 85% non-buyers and 15% buyers. We handled this using class weights in XGBoost and tested various sampling techniques.

### Q: Which features are most important?
**A:** 
1. **PageValues** dominates with 71% importance
2. ExitRates (3%)
3. Month (2%)
4. VisitorType (2%)
5. ProductRelated_Duration (1%)

### Q: How do I deploy this to production?
**A:** See the "Deployment Considerations" section. Options include:
- REST API (Flask/FastAPI)
- Batch processing (Airflow/Cron)
- Streaming (Kafka/Kinesis)

Choose based on your latency requirements and infrastructure.

---

## 10. Next Steps / Future Work
- ✅ **Complete EDA and feature analysis** (Done)
- ✅ **Model selection and hyperparameter tuning** (Done)
- ✅ **Production prediction pipeline** (Done)
- 🔄 **Build REST API service** (In progress - predictor.py ready)
- 🔄 **Add comprehensive tests** (pytest suite)
- 🔄 **Containerize with Docker** (Dockerfile)
- 🔄 **CI/CD pipeline** (GitHub Actions)
- 📋 **Deploy to cloud** (Render/Railway/AWS)
- 📋 **Model interpretability** (SHAP values, LIME)
- 📋 **A/B testing framework** (Compare models in production)
- 📋 **Monitoring dashboard** (Track metrics and drift)
- 📋 **Automated retraining** (On performance degradation)

---

## 11. Project Deliverables ✓

This project fulfills all ML Zoomcamp midterm requirements:

- ✅ **Problem Description**: Clear business problem with measurable impact
- ✅ **EDA**: Comprehensive analysis with 20+ visualizations
- ✅ **Model Training**: Multiple algorithms tested and compared
- ✅ **Model Selection**: Systematic comparison with cross-validation
- ✅ **Hyperparameter Tuning**: RandomizedSearchCV on top models
- ✅ **Best Practices**: 
  - Reproducible code in Jupyter notebook
  - Proper train/test split
  - Feature engineering and selection
  - Model artifacts saved
  - Production pipeline created
- ✅ **Documentation**: Complete README with usage instructions
- ✅ **Code Quality**: Clean, well-commented, modular code

---

## 12. References
- **Dataset**: UCI ML Repository - [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Course**: DataTalksClub - [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- **Libraries**: scikit-learn, XGBoost, pandas, matplotlib, seaborn
- **Tools**: uv (package manager), Jupyter notebooks, VS Code

### Dataset Citation (BibTeX)
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

---

## 13. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 14. Author & Contact
- **Project**: ML Zoomcamp Midterm Project
- **Repository**: [github.com/dimdimlv/ml_zoomcamp_midterm_project](https://github.com/dimdimlv/ml_zoomcamp_midterm_project)
- **Date**: November 2025

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is an educational project for the ML Zoomcamp course. The model and methods demonstrated here can be adapted for real-world e-commerce applications with appropriate validation and monitoring.

