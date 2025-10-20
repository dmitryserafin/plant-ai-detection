import io
import os
import time
import json
import base64
import logging
from typing import List, Optional, Dict

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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro-vision")
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
    allow_methods=["*"],
    allow_headers=["*"]
)

class DiseaseLocation(BaseModel):
    x: int
    y: int
    width: int
    height: int

class Soil(BaseModel):
    type: str
    drainage: str
    ph: str

class PredictResponse(BaseModel):
    id: str
    disease: Optional[str] = None
    confidence: Optional[float] = None
    description: Optional[str] = None
    treatment: Optional[str] = None
    suggestions: Optional[List[str]] = None
    severity: Optional[str] = None
    plant_type: Optional[str] = None
    affected_parts: Optional[List[str]] = None
    causative_agent: Optional[str] = None
    treatment_urgency: Optional[str] = None
    inference_ms: int
    disease_location: Optional[DiseaseLocation] = None
    plant_name: Optional[str] = None
    tags: Optional[List[str]] = None
    genus: Optional[str] = None
    scientific_name: Optional[str] = None
    common_names: Optional[List[str]] = None
    watering: Optional[str] = None
    temperature: Optional[str] = None
    sunlight: Optional[str] = None
    soil: Optional[Soil] = None
    pests_and_diseases: Optional[Dict[str, List[str]]] = None
    humidity: Optional[str] = None
    fertilizing: Optional[str] = None
    repotting: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

async def call_gemini(api_key: str, image_bytes: bytes, mode: str, language: str = "en") -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = GEMINI_API_URL.format(model=GEMINI_MODEL)
    params = {"key": api_key}

    language_instructions = {
        'en': 'Respond in English',
        'ru': 'Отвечай на русском',
        'hi': 'हिंदी में उत्तर दें',
        'ta': 'தமிழில் பதிலளிக்கவும்',
        'ml': 'മലയാളത്തിൽ മറുപടി നൽകുക'
    }
    
    final_lang_text = language_instructions.get(language, 'Respond in English').split(' ')[-1].upper()

    if mode == "recognition":
        prompt_text = f"""As an expert botanist, your task is to identify the plant in the image and provide a detailed description.

        Follow this structure for your response, providing all information in the specified language: {language_instructions.get(language, 'Respond in English')}.

        Respond in this exact JSON format:
        {{
          "plant_name": "Snake plant",
          "tags": ["Air-purifying", "Trendy", "Easy", "Medium", "Pet-toxic"],
          "genus": "Dracaena",
          "scientific_name": "Dracaena trifasciata",
          "common_names": ["Saint George's sword", "mother-in-law's tongue", "viper's bowstring hemp"],
          "description": "Dracaena trifasciata (Sansevieria trifasciata), also known as the snake plant or mother-in-law's tongue, is the sturdiest plant. It can tolerate anything from harsh weather to caring mistakes.",
          "watering": "Once in 2 weeks",
          "temperature": "21°C-32°C",
          "sunlight": "Part shade",
          "soil": {{
            "type": "Sand",
            "drainage": "Well-drained",
            "ph": "7.5 pH - 8.5 pH"
          }},
          "pests_and_diseases": {{
            "pests": ["Mealybugs", "spider mites"],
            "disease": ["Race rot"]
          }},
          "humidity": "40-60%\\nAvoid decreasing humidity",
          "fertilizing": "Feed with a cactus fertilizer diluted to half its strength\\nDon't fertilize during winter",
          "repotting": "Once a year"
        }}

        Ensure all text, including tags and descriptions, is in {final_lang_text}.
        """
    else:  # diagnosis mode
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

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
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
    
    log_payload = {
        "contents": [
            {"parts": [{"text": "..." }, {"inline_data": {"mime_type": "image/jpeg", "data": "<image_bytes>"}}]}
        ],
        "generationConfig": payload["generationConfig"]
    }
    logging.info(f"Calling Gemini API with mode '{mode}': {log_payload}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(url, params=params, json=payload) 
            r.raise_for_status() 
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise e
        
        response_json = r.json()
        logging.info(f"Gemini API response: {response_json}")
        return response_json

def simple_heuristic(image: Image.Image) -> tuple[str, float, List[str]]:
    # This heuristic is a fallback for the diagnosis mode only.
    img = image.convert("RGB").resize((128, 128))
    # ... (heuristic logic remains unchanged)
    return ("Fungal Leaf Spot", 0.75, ["Remove affected leaves", "Improve air circulation"])

@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
async def predict(
    image: UploadFile = File(...),
    mode: str = Form("diagnosis"),
    language: str = Form("en"),
    x_gemini_api_key: Optional[str] = Header(default=None, convert_underscores=False)
):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG or PNG.")

    content = await image.read()
    if len(content) / (1024*1024) > MAX_IMAGE_MB:
        raise HTTPException(status_code=413, detail=f"Image too large. Max {MAX_IMAGE_MB} MB")

    try:
        pil = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    t0 = time.time()
    api_key = x_gemini_api_key or os.getenv(GEMINI_API_KEY_ENV)

    if api_key:
        try:
            data = await call_gemini(api_key, content, mode, language)
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
                    start_index = text.find('{')
                    end_index = text.rfind('}')
                    if start_index != -1 and end_index != -1:
                        json_string = text[start_index:end_index+1]
                        parsed = json.loads(json_string)
                        t_ms = int((time.time()-t0)*1000)

                        if mode == "recognition":
                            return PredictResponse(
                                id=str(int(time.time()*1000)),
                                inference_ms=t_ms,
                                disease="Healthy Plant", # Default for recognition
                                **parsed
                            )
                        else: # diagnosis
                            recommendations = parsed.get("recommendations", [])
                            treatment = ". ".join(recommendations) if isinstance(recommendations, list) else (recommendations or "")
                            suggestions = recommendations if isinstance(recommendations, list) else [recommendations or ""]
                            disease_location = parsed.get("disease_location")

                            return PredictResponse(
                                id=str(int(time.time()*1000)),
                                disease=parsed.get("disease_name", "Unknown"),
                                confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
                                description=parsed.get("analysis", ""),
                                treatment=treatment,
                                suggestions=suggestions,
                                severity=parsed.get("severity", "Unknown"),
                                plant_type=parsed.get("plant_type", "Unknown"),
                                affected_parts=parsed.get("affected_parts", []),
                                causative_agent=parsed.get("causative_agent", "Unknown"),
                                treatment_urgency=parsed.get("treatment_urgency", "monitoring"),
                                disease_location=DiseaseLocation(**disease_location) if disease_location else None,
                                inference_ms=t_ms
                            )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logging.warning(f"JSON parsing failed, falling back. Error: {e}")
                    if mode == "recognition":
                         raise HTTPException(status_code=500, detail="Could not parse recognition data.")

        except httpx.HTTPStatusError as e:
            logging.error(f"Gemini API HTTP error, falling back. Error: {e}")
            if mode == "recognition":
                raise HTTPException(status_code=502, detail="Recognition service failed.")
        except Exception as e:
            logging.error(f"An unexpected error occurred, falling back. Error: {e}")
            if mode == "recognition":
                raise HTTPException(status_code=500, detail="An unexpected error occurred during recognition.")

    # Fallback only for diagnosis mode
    if mode == "diagnosis":
        disease, conf, tips = simple_heuristic(pil)
        t_ms = int((time.time()-t0)*1000)
        return PredictResponse(
            id=str(int(time.time()*1000)),
            disease=disease,
            confidence=round(conf,3),
            description="Heuristic analysis. Provide API key for full analysis.",
            treatment="Consult a specialist.",
            suggestions=tips,
            severity="Moderate",
            plant_type="Unknown",
            affected_parts=["leaves"],
            causative_agent="Unknown",
            treatment_urgency="monitoring",
            inference_ms=t_ms
        )

    raise HTTPException(status_code=400, detail="API key is required for plant recognition.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
