import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# --- IMPORT THE MIDDLEWARE ---
from fastapi.middleware.cors import CORSMiddleware


# --- Import Your Modular Validator Functions ---
# These imports will trigger the loading of your models ONCE when the app starts.
from template_validatorONNX import run_template_check
from face_validator import run_face_check
from field_validator import run_fields_check

# --- FastAPI Application ---
app = FastAPI(
    title="AI College ID Card Validator",
    description="Validates if an uploaded ID card is genuine or fake.",
)

# --- CORS Middleware Configuration ---
# ✅ CORRECTED: Defined origins clearly and removed duplicate middleware
origins = [
    "http://localhost",       # For general local development
    "http://localhost:3000",  # For the Dockerized frontend
    "http://127.0.0.1:3000",  # Another common local address
    # Add your production frontend URL here when you deploy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✨ ADDED: A simple health check endpoint for Docker Compose
@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- Pydantic Models for Input and Output ---
class IDCardRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., description="Base64 encoded string of the ID card image.")

class ValidationResponse(BaseModel):
    user_id: str
    validation_score: float = Field(..., ge=0, le=1)
    label: str
    status: str
    reason: str
    threshold: float

# --- Helper Function ---
def base64_to_cv2_image(base64_string: str):
    """Decodes a base64 string to an OpenCV image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 string or image format.")

# --- API Endpoint ---
@app.post("/validate-id", response_model=ValidationResponse)
def validate_id_card(request: IDCardRequest):
    """
    Receives a user ID and a base64 image, validates the ID card,
    and returns a detailed validation result.
    """
    APPROVE_THRESHOLD = 0.80
    REJECT_THRESHOLD = 0.50
    WEIGHTS = {
        "template": 0.5,
        "face": 0.3,
        "fields": 0.2
    }

    image = base64_to_cv2_image(request.image_base64)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please check format.")

    template_result = run_template_check(image)
    face_result = run_face_check(image)
    fields_result = run_fields_check(image)

    final_score = (
        template_result["score"] * WEIGHTS["template"] +
        face_result["score"] * WEIGHTS["face"] +
        fields_result["score"] * WEIGHTS["fields"]
    )
    
    reasons = []
    if template_result["score"] < 0.7:
        reasons.append("Template mismatch")
    if face_result["score"] < 0.6:
        reasons.append(f"Face validation failed ({face_result['details']})")
    if fields_result["score"] < 0.6:
        reasons.append(f"Low OCR confidence ({fields_result['details']})")

    if final_score >= APPROVE_THRESHOLD:
        status = "approved"
        label = "genuine"
        final_reason = "All checks passed with high confidence."
    elif final_score < REJECT_THRESHOLD:
        status = "rejected"
        label = "fake"
        final_reason = " and ".join(reasons) if reasons else "Overall confidence score is very low."
    else:
        status = "manual_review"
        label = "suspicious"
        final_reason = " and ".join(reasons) if reasons else "Indeterminate confidence score requires review."
    
    return ValidationResponse(
        user_id=request.user_id,
        validation_score=round(final_score, 2),
        label=label,
        status=status,
        reason=final_reason,
        threshold=APPROVE_THRESHOLD,
    )