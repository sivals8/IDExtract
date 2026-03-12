from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import pytesseract
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    return thresh

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Please upload a valid image file."}

    image_bytes = await file.read()
    processed = preprocess_image(image_bytes)
    raw_text = pytesseract.image_to_string(processed)

    
    for word in raw_text:
        if word == " 3":
            word == " + "
    
    split = raw_text.split('+')

    name = split[1]
    id_number = split[2]
    print(raw_text)


    return {
        "name": name,
        "id_number": id_number,
        "date_of_birth": None,
        "expiry_date": None,
        "raw_text": split,
    }