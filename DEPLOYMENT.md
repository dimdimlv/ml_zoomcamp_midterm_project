# Deployment Guide - REST API Service

This guide covers deploying the Online Shopper Purchase Intention prediction model as a REST API service.

## 📋 Table of Contents
1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment Options](#cloud-deployment-options)
4. [API Documentation](#api-documentation)
5. [Testing the API](#testing-the-api)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🚀 Local Deployment

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)
- All model artifacts in `models/` directory

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the API Server

```bash
# Start the server
python app.py

# Or use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

### Step 3: Test the API

```bash
# In a new terminal, run the test script
python test_api.py
```

Or visit the interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🐳 Docker Deployment

### Build the Docker Image

```bash
# Build the image
docker build -t online-shopper-api:latest .

# Verify the image was created
docker images | grep online-shopper-api
```

### Run the Container

```bash
# Run the container
docker run -d \
  --name shopper-api \
  -p 8000:8000 \
  online-shopper-api:latest

# Check container status
docker ps

# View logs
docker logs shopper-api

# Follow logs in real-time
docker logs -f shopper-api
```

### Test the Containerized API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction
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

### Stop and Remove Container

```bash
# Stop the container
docker stop shopper-api

# Remove the container
docker rm shopper-api

# Remove the image (optional)
docker rmi online-shopper-api:latest
```

---

## ☁️ Cloud Deployment Options

### Option 1: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p docker online-shopper-api --region us-east-1

# Create environment and deploy
eb create shopper-api-env

# Open in browser
eb open

# Update the deployment
eb deploy
```

### Option 2: Google Cloud Run

```bash
# Install gcloud CLI (if not already installed)
# See: https://cloud.google.com/sdk/docs/install

# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/online-shopper-api

# Deploy to Cloud Run
gcloud run deploy online-shopper-api \
  --image gcr.io/YOUR_PROJECT_ID/online-shopper-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 3: Azure Container Instances

```bash
# Install Azure CLI (if not already installed)
# See: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login to Azure
az login

# Create resource group
az group create --name shopper-api-rg --location eastus

# Create container registry
az acr create --resource-group shopper-api-rg \
  --name shopperapi --sku Basic

# Login to ACR
az acr login --name shopperapi

# Tag and push image
docker tag online-shopper-api:latest shopperapi.azurecr.io/online-shopper-api:latest
docker push shopperapi.azurecr.io/online-shopper-api:latest

# Deploy to ACI
az container create \
  --resource-group shopper-api-rg \
  --name shopper-api \
  --image shopperapi.azurecr.io/online-shopper-api:latest \
  --dns-name-label shopper-api-unique \
  --ports 8000
```

### Option 4: DigitalOcean App Platform

1. Push your code to GitHub
2. Go to DigitalOcean App Platform
3. Create new app from GitHub repository
4. Select Dockerfile deployment
5. Configure environment variables (if needed)
6. Deploy!

---

## 📚 API Documentation

### Endpoints

#### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: API information and available endpoints
- **Response**:
  ```json
  {
    "message": "Online Shopper Purchase Intention Predictor API",
    "version": "1.0.0",
    "endpoints": {...}
  }
  ```

#### 2. Health Check
- **URL**: `GET /health`
- **Description**: Check if service is healthy and model is loaded
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

#### 3. Single Prediction
- **URL**: `POST /predict`
- **Description**: Predict purchase intention for a single session
- **Request Body**:
  ```json
  {
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
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 1,
    "probability": 0.85,
    "confidence": 0.85,
    "label": "Revenue"
  }
  ```

#### 4. Batch Prediction
- **URL**: `POST /predict/batch`
- **Description**: Predict purchase intention for multiple sessions
- **Request Body**:
  ```json
  {
    "sessions": [
      {
        "Administrative": 0,
        ...
      },
      {
        "Administrative": 1,
        ...
      }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "predictions": [
      {
        "prediction": 1,
        "probability": 0.85,
        "confidence": 0.85,
        "label": "Revenue"
      },
      ...
    ],
    "count": 2
  }
  ```

---

## 🧪 Testing the API

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
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
        "Weekend": False
    }
)

print(response.json())
```

### Using cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_session.json
```

### Using the Test Script

```bash
python test_api.py
```

---

## 📊 Monitoring & Maintenance

### Logging

The application logs all requests and errors. To view logs:

```bash
# Docker logs
docker logs shopper-api

# Follow logs in real-time
docker logs -f shopper-api
```

### Performance Metrics

Monitor these key metrics:
- **Response Time**: Should be < 100ms for single predictions
- **Throughput**: Can handle 100+ requests per second
- **Error Rate**: Should be < 1%
- **CPU/Memory Usage**: Monitor container resources

### Health Monitoring

Set up automated health checks:

```bash
# Cron job to check health every 5 minutes
*/5 * * * * curl -f http://localhost:8000/health || alert_admin
```

### Updating the Model

To update the model without downtime:

1. Train and save new model artifacts
2. Build new Docker image with updated models
3. Deploy new version with rolling update
4. Verify with health check
5. Route traffic to new version

---

## 🔒 Security Considerations

### Production Recommendations

1. **Add Authentication**
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/predict")
   async def predict(session: SessionData, credentials: HTTPBearer = Depends(security)):
       # Validate token
       ...
   ```

2. **Rate Limiting**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/predict")
   @limiter.limit("100/minute")
   async def predict(...):
       ...
   ```

3. **HTTPS Only**: Use reverse proxy (nginx) or cloud load balancer
4. **Input Validation**: Already implemented with Pydantic
5. **Logging**: Log all requests for audit trail
6. **Environment Variables**: Store sensitive config in env vars

---

## 🎯 Performance Optimization

### For High Traffic

1. **Use Gunicorn with Multiple Workers**
   ```bash
   gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```

2. **Add Redis Caching** for repeated predictions
3. **Load Balancing**: Deploy multiple instances behind load balancer
4. **Auto-scaling**: Configure based on CPU/memory thresholds

---

## 📞 Support

For issues or questions:
- GitHub Issues: [Project Issues](https://github.com/dimdimlv/ml_zoomcamp_midterm_project/issues)
- Email: [Your contact email]

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
