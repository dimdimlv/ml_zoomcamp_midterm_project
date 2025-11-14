# Quick Start - REST API Deployment

This guide shows you how to quickly deploy the REST API using `uv`.

## Option 1: Local Development (Recommended for Testing)

```bash
# 1. Install web dependencies with uv
uv sync --group web

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Run the API server
python app.py

# 4. Test the API (in a new terminal)
python test_api.py

# 5. Open interactive docs in browser
open http://localhost:8000/docs
```

## Option 2: Using uv run (No activation needed)

```bash
# 1. Install web dependencies
uv sync --group web

# 2. Run the API directly with uv
uv run python app.py

# 3. In another terminal, test the API
uv run python test_api.py
```

## Option 3: Docker (Production Ready)

```bash
# 1. Build and run with docker-compose
docker-compose up -d

# 2. View logs
docker-compose logs -f

# 3. Stop the service
docker-compose down
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Make a Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 15,
    "ProductRelated_Duration": 800.0,
    "BounceRates": 0.01,
    "ExitRates": 0.02,
    "PageValues": 25.5,
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

### Or Run the Test Script
```bash
# If environment is activated
python test_api.py

# Or with uv
uv run python test_api.py
```

## Interactive Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly from the browser!

## Troubleshooting

### Port 8000 already in use
```bash
# Find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9

# Or run on a different port
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Dependencies not found
```bash
# Make sure web dependencies are installed
uv sync --group web

# Check installed packages
uv pip list | grep -E 'fastapi|uvicorn|pydantic'
```

### Model not loading
```bash
# Verify model files exist
ls -lh models/

# Should see: final_model.pkl, scaler.pkl, label_encoders.pkl, feature_names.json
```

## Next Steps

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Cloud deployment options (AWS, Google Cloud, Azure, etc.)
- Production configuration
- Monitoring and scaling
- Security considerations
