import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageEnhance
from transformers import pipeline

app = FastAPI(
    title="Kisan AI - Precision Crop Diagnostic Core",
    description="Vision AI diagnosis with confidence filtering and out-of-domain detection.",
    version="3.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Agricultural Vision Model...")
MODEL_ID = "kimcomehome/plantvillage-vit-leaf-disease"
classifier = pipeline("image-classification", model=MODEL_ID)
print("Universal Crop Vision Engine Loaded Successfully!")

CONFIDENCE_THRESHOLD_MIN = 35.0
CONFIDENCE_THRESHOLD_HIGH = 60.0

def preprocess_leaf(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.15)

@app.get("/")
def root():
    return {"status": "online", "model": "ViT Crop Disease Diagnostic Engine"}

@app.post("/api/v1/predict/leaf")
async def detect_crop_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_leaf(raw_image)

        predictions = classifier(processed_image, top_k=3)
        top_match = predictions[0]
        confidence = round(top_match["score"] * 100, 2)

        def parse_label(label: str):
            clean = label.replace("___", " - ").replace("_", " ")
            if " - " in clean:
                parts = clean.split(" - ")
                crop_raw = parts[0].strip().replace("Pepper, bell", "Chilli / Pepper").title()
                disease_raw = parts[1].strip().title()
                return crop_raw, disease_raw
            return "Plant / Crop", clean.strip().title()

        crop, disease = parse_label(top_match["label"])

        # Guardrail: Catch low confidence / unsupported crops
        if confidence < CONFIDENCE_THRESHOLD_MIN:
            return {
                "success": False,
                "status": "UNRECOGNIZED_OR_LOW_CONFIDENCE",
                "top_guess": f"{crop} ({disease})",
                "confidence_score": str(confidence) + "%",
                "message": "The crop leaf could not be identified with certainty. This usually happens if the leaf is Paddy/Rice, Wheat, or the photo is blurry.",
                "guidance": "Supported crops in this model: Tomato, Potato, Corn, Pepper/Chilli, Apple, Grape, Strawberry."
            }

        return {
            "success": True,
            "crop": crop,
            "disease_detected": disease,
            "confidence_score": str(confidence) + "%",
            "certainty_level": "HIGH" if confidence >= CONFIDENCE_THRESHOLD_HIGH else "MODERATE",
            "top_possibilities": [
                {
                    "crop": parse_label(p["label"])[0],
                    "disease": parse_label(p["label"])[1],
                    "confidence": str(round(p["score"] * 100, 2)) + "%"
                }
                for p in predictions
            ]
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Diagnosis error: {str(error)}")
