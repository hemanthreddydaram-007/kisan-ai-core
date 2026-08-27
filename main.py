import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline

app = FastAPI(title="Kisan AI Core")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_ID = "kimcomehome/plantvillage-vit-leaf-disease"
print("Loading AI model pipeline...")
classifier = pipeline("image-classification", model=MODEL_ID)
print("Crop AI Model loaded and ready!")

@app.get("/")
def health(): return {"status": "online", "message": "Crop Advisory Ready"}

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"): raise HTTPException(status_code=400, detail="Invalid image")
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        preds = classifier(img)
        top = preds[0]
        raw = top["label"]
        crop, dis = raw.split("___") if "___" in raw else ("Plant", raw)
        return {"success": True, "crop": crop, "disease": dis.replace("_", " "), "confidence": round(top["score"]*100, 2), "top_3": [{"label": p["label"].replace("___", " - ").replace("_", " "), "confidence": round(p["score"]*100, 2)} for p in preds[:3]]}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
