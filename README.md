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
### Local Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data-path data/online_shoppers_intention.csv
uvicorn src.serve:app --host 0.0.0.0 --port 9696 --reload
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

## 7. Next Steps / Future Work
- Try **XGBoost** or **CatBoost** models.
- Use **SMOTE** or class weighting for imbalance.
- Add **SHAP** or **LIME** interpretability.
- Deploy to **Render / Railway**.
- Add automated testing (pytest + REST client).

---

## 8. References
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

