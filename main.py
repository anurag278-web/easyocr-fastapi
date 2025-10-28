from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import easyocr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EasyOCR FastAPI Server",
              description="OCR server using EasyOCR (Render optimized)",
              version="1.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


def build_reader(requested_langs, gpu=False):
    # Default to English only
    langs = [lang.strip().lower() for lang in (requested_langs or ["en"])]
    if "en" not in langs:
        langs.append("en")

    try:
        reader = easyocr.Reader(langs, gpu=gpu)
        logging.info(f"✅ Initialized OCR for: {langs}")
        return reader, langs
    except Exception as e:
        logging.warning(f"⚠️ Error initializing EasyOCR ({e}), fallback to English")
        return easyocr.Reader(["en"], gpu=gpu), ["en"]


@app.post("/ocr", response_model=OCRResult)
async def ocr_endpoint(file: UploadFile = File(...), languages: Optional[str] = None):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    image = validate_image_bytes(file_bytes)

    requested_langs = [l.strip() for l in languages.split(",")] if languages else ["en"]
    reader, used_langs = build_reader(requested_langs=requested_langs, gpu=False)

    img_np = np.array(image)
    results = reader.readtext(img_np, detail=1)

    text_boxes = [
        {"bbox": [[float(x), float(y)] for x, y in bbox], "text": text, "confidence": float(conf)}
        for bbox, text, conf in results
    ]

    width, height = image.size
    return JSONResponse(content={
        "img_width": width,
        "img_height": height,
        "languages_used": used_langs,
        "text_boxes": text_boxes
    })


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
