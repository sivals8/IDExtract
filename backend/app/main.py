from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import pytesseract
import io
import re

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

def parse_fields(raw_text: str):
    result = {
        "name": None,
        "designation": None,
        "id_number": None,
        "issued_date": None,
        "expiry_date": None,
    }

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    #first clean the output
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", line).strip()
        cleaned_lines.append(cleaned)
    
    #pattern to identify dates (currently only matches my sample format dd-mmm-yyyy)
    date_pattern = r"\d{1,2}\s*-\s*[A-Za-z]{3}\s*-\s*\d{4}"

    for line in cleaned_lines:
        lower = line.lower()

        print("LINE:", line)
        print("LOWER:", line.lower())

        if "issued" in lower or "joined" in lower:
            match = re.search(date_pattern, line) 

            if match:
                result["issued_date"] = re.sub(r"\s+", "", match.group()).replace("/", "-")
            
        if "expires" in lower:
            match = re.search(date_pattern, line)
            if match:
                result["expiry_date"] = re.sub(r"\s+", "", match.group()).replace("/", "-")

        
    #ID number recognition
    
    for line in cleaned_lines:
        if "id" in line.lower():
            match = re.search(r"\d{6,}", line)
            if match:
                result["id_number"] = match.group()
                break

    #fallback if the 'ID' is missed
    if result["id_number"] is None:
        for line in cleaned_lines:
            match = re.search(r"\d{6,}", line)
            if match:
                result["id_number"] = match.group()
                break

    #name and role/title
    candidate_lines = []
    for line in cleaned_lines:
        lower = line.lower()

        if "issued" in lower or "expire" in lower or "id" in lower:
            continue
        if re.search(r"\d{4,}", line):
            continue
        if len(line) < 3:
            continue
        candidate_lines.append(line)

    if len(candidate_lines) >= 1:
        result["name"] = candidate_lines[0]

    if len(candidate_lines) >= 2:
        result["designation"] = candidate_lines[1]


    print("RAW TEXT:", repr(raw_text))
    print("RAW TEXT LENGTH:", len(raw_text))

    return result




@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Please upload a valid image file."}

    image_bytes = await file.read()
    processed = preprocess_image(image_bytes)
    raw_text = pytesseract.image_to_string(processed)

    parsed = parse_fields(raw_text)


    return {
        "name": parsed["name"],
        "designation": parsed["designation"],
        "id_number": parsed["id_number"],
        "issued_date": parsed["issued_date"],
        "expiry_date": parsed["expiry_date"],
        "raw_text": raw_text,
    }