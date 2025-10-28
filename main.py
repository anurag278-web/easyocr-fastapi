from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from PIL import Image
import io
import easyocr
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EasyOCR FastAPI Server",
              description="OCR server using EasyOCR (multiple Indian languages)",
              version="1.0")

# Enable CORS for any origin (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default languages to try to load (English + common Indian languages)
DEFAULT_LANGS = ["en", "hi", "ta", "te", "bn"]


class TextBox(BaseModel):
    bbox: List[List[float]]
    text: str
    confidence: float


class OCRResult(BaseModel):
    img_width: int
    img_height: int
    languages_used: List[str]
    text_boxes: List[TextBox]


def validate_image_bytes(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image or is corrupted.")


def build_reader(requested_langs, gpu=False):
    # convert to lowercase and remove spaces
    langs = [lang.strip().lower() for lang in requested_langs]

    # ✅ EasyOCR rule: regional languages like Telugu, Hindi, Tamil must include 'en'
    safe_langs = []
    for lang in langs:
        if lang != "en":
            safe_langs.append(lang)
    if "en" not in safe_langs:
        safe_langs.append("en")

    try:
        reader = easyocr.Reader(safe_langs, gpu=gpu)
        logging.info(f"Initialized OCR for: {safe_langs}")
        return reader, safe_langs
    except ValueError as e:
        # Handle cases like Telugu or Tamil incompatibility
        logging.error(f"Error initializing EasyOCR: {e}")
        # fallback to English only
        reader = easyocr.Reader(['en'], gpu=gpu)
        return reader, ['en']

@app.post("/ocr", response_model=OCRResult)
async def ocr_endpoint(file: UploadFile = File(...), languages: Optional[str] = None):
    """
    Accepts an image file (multipart/form-data) and optional comma-separated languages query param.
    Example: POST /ocr?languages=en,hi,ta
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file sent in request.")

    content_type = file.content_type.lower() if file.content_type else ""
    if not any(ct in content_type for ct in ["image/jpeg", "image/png", "image/jpg"]):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}. Please upload JPEG or PNG images.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = validate_image_bytes(file_bytes)

    # Parse languages query param (comma separated)
    requested_langs = None
    if languages:
        requested_langs = [l.strip() for l in languages.split(",") if l.strip()]

    # Build EasyOCR reader (gpu=False -> CPU)
    reader, used_langs = build_reader(requested_langs=requested_langs, gpu=False)

    # Convert PIL image to numpy array expected by EasyOCR
    import numpy as np
    img_np = np.array(image)

    try:
        results = reader.readtext(img_np, detail=1)
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        raise HTTPException(status_code=500, detail=f"Error during OCR processing: {str(e)}")

    text_boxes = []
    for bbox, text, conf in results:
        # bbox is typically a list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text_boxes.append({
            "bbox": [[float(p[0]), float(p[1])] for p in bbox],
            "text": text,
            "confidence": float(conf)
        })

    width, height = image.size

    response = {
        "img_width": width,
        "img_height": height,
        "languages_used": used_langs,
        "text_boxes": text_boxes,
    }

    return JSONResponse(status_code=200, content=response)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# If you want to run with `python main.py` for local dev convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

@app.on_event("startup")
def preload_models():
    import easyocr
    import logging
    try:
        logging.info("🔄 Preloading OCR models...")
        easyocr.Reader(["en", "hi", "ta", "te", "bn"], gpu=False)
        logging.info("✅ OCR models preloaded successfully!")
    except Exception as e:
        logging.warning(f"⚠️ Failed to preload OCR models: {e}")
