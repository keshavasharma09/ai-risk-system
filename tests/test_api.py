from fastapi.testclient import TestClient
from api.main import app 

client = TestClient(app)

def test_health_endpoint():
    """Test that the API is up and running successfully"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running!"}