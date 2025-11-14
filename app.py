"""
FastAPI REST API for Online Shopper Purchase Intention Prediction

This service provides endpoints to predict whether a website visitor
will make a purchase during an online shopping session.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import uvicorn
from src.predictor import OnlineShopperPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Online Shopper Purchase Intention Predictor",
    description="Predict whether a visitor will make a purchase based on session data",
    version="1.0.0",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loaded once at startup)
predictor = None


class SessionData(BaseModel):
    """Input schema for a single shopping session"""
    
    Administrative: int = Field(..., ge=0, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(..., ge=0, description="Time spent on administrative pages")
    Informational: int = Field(..., ge=0, description="Number of informational pages visited")
    Informational_Duration: float = Field(..., ge=0, description="Time spent on informational pages")
    ProductRelated: int = Field(..., ge=0, description="Number of product-related pages visited")
    ProductRelated_Duration: float = Field(..., ge=0, description="Time spent on product-related pages")
    BounceRates: float = Field(..., ge=0, le=1, description="Bounce rate of pages visited")
    ExitRates: float = Field(..., ge=0, le=1, description="Exit rate of pages visited")
    PageValues: float = Field(..., ge=0, description="Average page value")
    SpecialDay: float = Field(..., ge=0, le=1, description="Proximity to special day (0-1)")
    Month: str = Field(..., description="Month of the visit (e.g., 'Jan', 'Feb', 'Mar')")
    OperatingSystems: int = Field(..., ge=1, description="Operating system ID")
    Browser: int = Field(..., ge=1, description="Browser ID")
    Region: int = Field(..., ge=1, description="Geographic region ID")
    TrafficType: int = Field(..., ge=1, description="Traffic type ID")
    VisitorType: str = Field(..., description="Type of visitor (e.g., 'Returning_Visitor', 'New_Visitor')")
    Weekend: bool = Field(..., description="Whether visit occurred on weekend")

    class Config:
        json_schema_extra = {
            "example": {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 1,
                "ProductRelated_Duration": 0.0,
                "BounceRates": 0.2,
                "ExitRates": 0.2,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "Feb",
                "OperatingSystems": 1,
                "Browser": 1,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "Returning_Visitor",
                "Weekend": False
            }
        }


class BatchSessionData(BaseModel):
    """Input schema for batch predictions"""
    sessions: List[SessionData]


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    prediction: int = Field(..., description="Binary prediction (0=No Purchase, 1=Purchase)")
    probability: float = Field(..., description="Probability of purchase (0-1)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    label: str = Field(..., description="Human-readable prediction label")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global predictor
    try:
        predictor = OnlineShopperPredictor()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Online Shopper Purchase Intention Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check if the service is healthy and model is loaded"""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(session: SessionData):
    """
    Predict purchase intention for a single shopping session
    
    Args:
        session: SessionData object containing visitor session features
    
    Returns:
        PredictionResponse with prediction, probability, confidence, and label
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([session.dict()])
        
        # Convert column names to lowercase to match model training
        input_data.columns = input_data.columns.str.lower()
        
        # Make prediction
        result = predictor.predict_with_confidence(input_data)
        
        return {
            "prediction": int(result['prediction'].iloc[0]),
            "probability": float(result['probability'].iloc[0]),
            "confidence": float(result['confidence'].iloc[0]),
            "label": result['label'].iloc[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchSessionData):
    """
    Predict purchase intention for multiple shopping sessions
    
    Args:
        batch: BatchSessionData containing a list of sessions
    
    Returns:
        BatchPredictionResponse with predictions for all sessions
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([session.dict() for session in batch.sessions])
        
        # Convert column names to lowercase to match model training
        input_data.columns = input_data.columns.str.lower()
        
        # Make predictions
        results = predictor.predict_with_confidence(input_data)
        
        predictions = [
            {
                "prediction": int(row['prediction']),
                "probability": float(row['probability']),
                "confidence": float(row['confidence']),
                "label": row['label']
            }
            for _, row in results.iterrows()
        ]
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
