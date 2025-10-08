import io
import os
import time
import base64
import logging
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import httpx
from dotenv import load_dotenv

APP_TITLE = "PlantAI Backend"
APP_VERSION = "0.1.0"

# Environment config
# Load .env if present
load_dotenv()
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "8"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"]
    ,allow_headers=["*"]
)

class DiseaseLocation(BaseModel):
    x: int
    y: int
    width: int
    height: int

class PredictResponse(BaseModel):
    id: str
    disease: str
    confidence: float
    description: str
    treatment: str
    suggestions: List[str]
    severity: str
    plant_type: str
    affected_parts: List[str]
    causative_agent: str
    treatment_urgency: str
    inference_ms: int
    disease_location: Optional[DiseaseLocation] = None

class ErrorResponse(BaseModel):
    detail: str

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

async def call_gemini(api_key: str, image_bytes: bytes, language: str = "en") -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = GEMINI_API_URL.format(model=GEMINI_MODEL)
    params = {"key": api_key}

    language_instructions = {
        'en': 'Respond in English',
        'hi': 'हिंदी में उत्तर दें',
        'ta': 'தமிழில் பதிலளிக்கவும்',
        'ml': 'മലയാളത്തിൽ മറുപടി നൽകുക'
    }
    
    # Determine the language for the final instruction text
    final_lang_text = 'ENGLISH'
    if language == 'hi':
        final_lang_text = 'HINDI'
    elif language == 'ta':
        final_lang_text = 'TAMIL'
    elif language == 'ml':
        final_lang_text = 'MALAYALAM'

    prompt_text = f"""You are an expert plant pathologist with advanced knowledge in agricultural sciences, botany, and plant disease diagnosis. Analyze this plant image with the precision of a professional laboratory assessment.

ANALYSIS FRAMEWORK:
1. VISUAL EXAMINATION: Examine leaf morphology, coloration patterns, lesion characteristics, growth abnormalities, and environmental stress indicators
2. SYMPTOM IDENTIFICATION: Identify primary and secondary symptoms including chlorosis, necrosis, wilting, stunting, distortion, and pathogen signs
3. DIFFERENTIAL DIAGNOSIS: Consider multiple potential causes including fungal, bacterial, viral, nutritional, environmental, and pest-related factors
4. CONFIDENCE ASSESSMENT: Base confidence on symptom clarity, image quality, diagnostic specificity, and elimination of alternative causes

DIAGNOSTIC CRITERIA:
- Fungal diseases: Look for spores, mycelium, fruiting bodies, characteristic lesion patterns
- Bacterial diseases: Check for water-soaked lesions, bacterial ooze, systemic symptoms
- Viral diseases: Examine for mosaic patterns, ring spots, yellowing, stunting
- Nutritional deficiencies: Assess chlorosis patterns, leaf positioning, uniform vs. localized symptoms
- Environmental stress: Consider light conditions, water stress, temperature damage
- Pest damage: Look for feeding patterns, egg masses, insect presence

CRITICAL: {language_instructions.get(language, language_instructions['en'])}.

ALL FIELDS INCLUDING DISEASE NAMES, DESCRIPTIONS, RECOMMENDATIONS, AND TECHNICAL TERMS MUST BE IN THE SPECIFIED LANGUAGE.

IMAGE COORDINATE SYSTEM: The top-left corner is (0, 0). The bottom-right corner is (width, height). Provide coordinates for a single bounding box that encloses the most representative symptom.

Respond in this exact JSON format with all content in the specified language:
{{
  "disease_name": "specific disease name with scientific classification or 'Healthy Plant' in the target language",
  "confidence": 0.85,
  "analysis": "comprehensive explanation including symptoms observed, affected plant parts, disease progression stage, and reasoning for diagnosis in the target language",
  "recommendations": ["immediate treatment steps in target language", "preventive measures in target language", "monitoring guidelines in target language", "environmental modifications in target language", "follow-up actions in target language"],
  "severity": "Low/Moderate/High/Critical in target language",
  "plant_type": "identified plant species or family if determinable in target language",
  "affected_parts": ["leaves", "stems", "roots", "flowers", "fruits"] in target language,
  "causative_agent": "fungal/bacterial/viral/nutritional/environmental/pest in target language",
  "treatment_urgency": "immediate/within_week/routine_care/monitoring in target language",
  "disease_location": {{ "x": 120, "y": 250, "width": 80, "height": 100 }}
}}

CONFIDENCE SCORING:
- 0.9-1.0: Clear, unambiguous symptoms with high diagnostic certainty
- 0.7-0.89: Strong evidence with minor uncertainty or image limitations
- 0.5-0.69: Moderate confidence with some differential diagnosis needed
- 0.3-0.49: Low confidence due to early symptoms or image quality issues
- 0.1-0.29: Very uncertain, requires additional examination

Be scientifically accurate and provide actionable, safe recommendations. If multiple conditions are possible, mention the most likely primary diagnosis. Remember: ALL TEXT MUST BE IN {final_lang_text}.
"""

    # This is the actual payload that will be sent
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 0.8,
            "maxOutputTokens": 8192
        }
    }
    
    # Create the log payload separately for logging purposes
    log_payload = {
        "contents": [
            {
                "parts": [
                    {"text": "..." },
                    {"inline_data": {"mime_type": "image/jpeg", "data": "<image_bytes>"}}
                ]
            }
        ],
        "generationConfig": payload["generationConfig"]
    }
    logging.info(f"Calling Gemini API: {log_payload}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Use the original payload with the real base64 data
            r = await client.post(url, params=params, json=payload) 
            r.raise_for_status() 
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise e
        
        response_json = r.json()
        logging.info(f"Gemini API response: {response_json}")
        return response_json


def simple_heuristic(image: Image.Image) -> tuple[str, float, List[str]]:
    # Enhanced heuristic with multiple factors for realistic confidence
    img = image.convert("RGB").resize((128, 128))
    pixels = list(img.getdata())
    
    if not pixels:
        return ("Unknown", 0.0, ["Retake photo in better light."])
    
    total = len(pixels)
    
    # Color analysis
    greens = sum(1 for r,g,b in pixels if g > r+10 and g > b+10 and g > 80)
    browns = sum(1 for r,g,b in pixels if r > g+5 and r > b+5 and r > 100)
    yellows = sum(1 for r,g,b in pixels if r > 150 and g > 150 and b < 100)
    blacks = sum(1 for r,g,b in pixels if r < 50 and g < 50 and b < 50)
    
    # Brightness and contrast analysis
    brightness_vals = [0.299*r + 0.587*g + 0.114*b for r,g,b in pixels]
    avg_brightness = sum(brightness_vals) / total
    brightness_std = (sum((b - avg_brightness)**2 for b in brightness_vals) / total) ** 0.5
    
    # Edge detection simulation (contrast between adjacent pixels)
    edge_count = 0
    for i in range(0, len(pixels)-129, 128):  # Row by row
        for j in range(127):  # Adjacent pixels in row
            if abs(sum(pixels[i+j]) - sum(pixels[i+j+1])) > 100:
                edge_count += 1
    
    edge_density = edge_count / (127 * (total // 128)) if total >= 128 else 0
    
    # Calculate ratios
    green_ratio = greens / total
    brown_ratio = browns / total
    yellow_ratio = yellows / total
    black_ratio = blacks / total
    
    # Decision logic with confidence based on multiple factors
    if green_ratio > 0.4 and brown_ratio < 0.1 and yellow_ratio < 0.05:
        # Healthy plant indicators
        confidence = min(0.85, 0.6 + green_ratio * 0.3 + (brightness_std / 50) * 0.1)
        return ("Healthy", confidence, ["Continue current care routine", "Monitor for changes"])
    
    elif yellow_ratio > 0.15 or (yellow_ratio > 0.08 and brown_ratio > 0.05):
        # Yellowing suggests nutrient deficiency or disease
        confidence = min(0.82, 0.5 + yellow_ratio * 1.5 + brown_ratio * 0.8)
        return ("Leaf Yellowing (Nutrient Deficiency)", confidence, 
                ["Check nitrogen levels", "Improve drainage", "Reduce watering frequency"])
    
    elif brown_ratio > 0.2 or black_ratio > 0.1:
        # Brown/black spots suggest fungal issues
        confidence = min(0.78, 0.45 + brown_ratio * 1.2 + black_ratio * 1.5)
        return ("Fungal Leaf Spot", confidence, 
                ["Remove affected leaves", "Improve air circulation", "Apply fungicide if severe"])
    
    elif edge_density > 0.3 and brightness_std > 40:
        # High contrast/edges might indicate pest damage or disease patterns
        confidence = min(0.65, 0.4 + edge_density * 0.5)
        return ("Possible Pest Damage", confidence, 
                ["Inspect for insects", "Check undersides of leaves", "Consider organic pesticide"])
    
    elif avg_brightness < 80:
        # Very dark image
        confidence = 0.25
        return ("Poor Image Quality", confidence, 
                ["Retake in better lighting", "Ensure leaf is clearly visible"])
    
    else:
        # Uncertain case
        confidence = 0.35 + green_ratio * 0.2
        return ("Early Stage Stress", confidence, 
                ["Monitor closely", "Check watering schedule", "Ensure adequate light"])

@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
async def predict(
    image: UploadFile = File(...),
    language: str = Form("en"),
    x_gemini_api_key: Optional[str] = Header(default=None, convert_underscores=False)
):
    # Validate image
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG or PNG.")

    content = await image.read()
    size_mb = len(content) / (1024*1024)
    if size_mb > MAX_IMAGE_MB:
        raise HTTPException(status_code=413, detail=f"Image too large. Max {MAX_IMAGE_MB} MB")

    # Load PIL image for heuristic fallback
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    t0 = time.time()

    # Prefer header key; fall back to env key
    api_key = x_gemini_api_key or os.getenv(GEMINI_API_KEY_ENV)

    if api_key:
        try:
            data = await call_gemini(api_key, content, language)
            text = ""
            if isinstance(data, dict):
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "text" in p:
                            text += p["text"]
            
            if text:
                try:
                    # Find the start and end of the JSON object
                    start_index = text.find('{')
                    end_index = text.rfind('}')
                    
                    if start_index != -1 and end_index != -1:
                        import json
                        json_string = text[start_index:end_index+1]
                        parsed = json.loads(json_string)
                        
                        recommendations = parsed.get("recommendations", [])
                        treatment = ". ".join(recommendations) if isinstance(recommendations, list) else (recommendations or "No recommendations available")
                        suggestions = recommendations if isinstance(recommendations, list) else [recommendations or "No recommendations available"]

                        t_ms = int((time.time()-t0)*1000)
                        disease_location = parsed.get("disease_location")

                        t_ms = int((time.time()-t0)*1000)
                        return PredictResponse(
                            id=str(int(time.time()*1000)),
                            disease=parsed.get("disease_name", "Unknown"),
                            confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
                            description=parsed.get("analysis", "No analysis available"),
                            treatment=treatment,
                            suggestions=suggestions,
                            severity=parsed.get("severity", "Unknown"),
                            plant_type=parsed.get("plant_type", "Unknown plant"),
                            affected_parts=parsed.get("affected_parts", []),
                            causative_agent=parsed.get("causative_agent", "Unknown"),
                            treatment_urgency=parsed.get("treatment_urgency", "monitoring"),
                            disease_location=DiseaseLocation(**disease_location) if disease_location else None,
                            inference_ms=t_ms
                        )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logging.warning(f"JSON parsing failed, falling back to heuristic. Error: {e}")

        except httpx.HTTPStatusError as e:
            logging.error(f"Gemini API HTTP error, falling back to heuristic. Error: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred, falling back to heuristic. Error: {e}")

    # Fallback to heuristic if API key is missing, API call fails, or parsing fails
    disease, conf, tips = simple_heuristic(pil)
    t_ms = int((time.time()-t0)*1000)

    # The heuristic response needs to be adapted to the new PredictResponse model
    return PredictResponse(
        id=str(int(time.time()*1000)),
        disease=disease,
        confidence=round(conf,3),
        description=f"Heuristic analysis suggests possible {disease.lower()}. For a full analysis, please provide an API key.",
        treatment="Consult with a plant specialist for accurate diagnosis and treatment recommendations.",
        suggestions=tips,
        severity="Moderate",
        plant_type="Unknown plant",
        affected_parts=["leaves"],
        causative_agent="Unknown",
        treatment_urgency="monitoring",
        inference_ms=t_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
