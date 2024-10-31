from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_caption():
    with open("test_image.jpg", "rb") as image_file:
        response = client.post("/generate-caption/", files={"file": image_file})
    
    assert response.status_code == 200
    assert "caption" in response.json()
    assert isinstance(response.json()["caption"], str)
