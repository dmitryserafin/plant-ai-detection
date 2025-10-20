import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch, MagicMock
import os
import io
from PIL import Image

# Set a dummy API key for testing
os.environ["GEMINI_API_KEY"] = "test_api_key"

client = TestClient(app)

@pytest.fixture
def mock_httpx_client():
    with patch("app.httpx.AsyncClient") as mock_client:
        yield mock_client

def create_dummy_image():
    """Creates a simple, valid JPEG image in memory."""
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_predict_recognition_success(mock_httpx_client):
    # Mock the Gemini API response for recognition
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '''
```json
{
  "plant_name": "Snake plant",
  "tags": ["Air-purifying", "Trendy"],
  "genus": "Dracaena",
  "scientific_name": "Dracaena trifasciata",
  "common_names": ["mother-in-law's tongue"],
  "description": "A popular houseplant.",
  "watering": "Once every 2 weeks",
  "temperature": "18°C-35°C",
  "sunlight": "Indirect light",
  "soil": {
    "type": "Sandy",
    "drainage": "Well-drained",
    "ph": "6.0-7.5"
  },
  "pests_and_diseases": {
    "pests": ["Mealybugs"],
    "disease": ["Root rot"]
  },
  "humidity": "40-50%",
  "fertilizing": "Monthly during growing season",
  "repotting": "Every 2-3 years"
}
```
'''
                        }
                    ]
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_httpx_client.return_value.__aenter__.return_value.post = mock_post

    # Use a valid dummy image
    dummy_image = create_dummy_image()
    response = client.post(
        "/predict",
        files={"image": ("test_image.jpg", dummy_image, "image/jpeg")},
        data={"mode": "recognition", "language": "en"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["plant_name"] == "Snake plant"
    assert data["scientific_name"] == "Dracaena trifasciata"
    assert "mother-in-law's tongue" in data["common_names"]
    assert data["soil"]["type"] == "Sandy"

def test_predict_recognition_api_key_missing():
    # Unset the API key for this test
    original_key = os.environ.pop("GEMINI_API_KEY", None)

    dummy_image = create_dummy_image()
    response = client.post(
        "/predict",
        files={"image": ("test_image.jpg", dummy_image, "image/jpeg")},
        data={"mode": "recognition", "language": "en"}
    )

    # Restore the API key
    if original_key:
        os.environ["GEMINI_API_KEY"] = original_key

    assert response.status_code == 400
    assert response.json()["detail"] == "API key is required for plant recognition."
