"""
Test script for the Online Shopper Purchase Intention API

Run this script to test the API endpoints locally.
"""

import requests
import json

# Base URL - change if deployed elsewhere
BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint"""
    print("\n🔍 Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✓ Health check passed")


def test_root():
    """Test the root endpoint"""
    print("\n🔍 Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✓ Root endpoint passed")


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🔍 Testing Single Prediction...")
    
    # Example session data - High intent visitor
    session_data = {
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
    
    response = requests.post(f"{BASE_URL}/predict", json=session_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert "prediction" in result
    assert "probability" in result
    assert "confidence" in result
    assert "label" in result
    print("✓ Single prediction passed")
    
    return result


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔍 Testing Batch Prediction...")
    
    # Example batch data
    batch_data = {
        "sessions": [
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
                "Weekend": False
            },
            {
                "Administrative": 1,
                "Administrative_Duration": 10.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 1,
                "ProductRelated_Duration": 5.0,
                "BounceRates": 0.2,
                "ExitRates": 0.2,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "Feb",
                "OperatingSystems": 1,
                "Browser": 1,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "New_Visitor",
                "Weekend": False
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert "predictions" in result
    assert "count" in result
    assert result["count"] == 2
    print("✓ Batch prediction passed")
    
    return result


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Starting API Tests")
    print("=" * 60)
    
    try:
        test_root()
        test_health()
        result = test_single_prediction()
        test_batch_prediction()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
        print("\n📊 Example Prediction Result:")
        print(f"  Prediction: {result['label']}")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running with: python app.py")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
