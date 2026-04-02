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

# before preprocessing, want to rectify card first by 
#1. finding the boundary of the card vs the surface its on
#2. warp card so it becomes 2D (removes any angling from the way the pic is took)
def load_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def find_card_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

    return None


def rectify_card(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    card_contour = find_card_contour(edges)

    if card_contour is None:
        return image_bgr

    warped = four_point_transform(image_bgr, card_contour)
    return warped


# preprpcessing image so ocr can understand text better
def preprocess_image(image_bgr):
    enlarged = cv2.resize(
        image_bgr,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

@app.get("/")
def root():
    return {"message": "Backend is running"}

def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    line = line.replace("IDNo", "ID No")
    line = line.replace("IDNO", "ID No")
    line = line.replace("IdNo", "ID No")
    return line


def clean_person_text(text: str):
    text = text.strip()
    text = re.sub(r"^[^A-Za-z]+", "", text)
    text = re.sub(r"[^A-Za-z\s\-.]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 4:
        return None
    if not re.search(r"[A-Za-z]", text):
        return None

    weird_count = len(re.findall(r"[^A-Za-z\s\-.]", text))
    if weird_count > 2:
        return None

    return text


def normalize_date(text: str) -> str:
    # convert multiple spaces into single space
    text = re.sub(r"\s+", " ", text).strip()

    # normalize separators
    text = text.replace("/", "-")

    # fix cases like "Nov- 2020" or "Nov -2020"
    text = re.sub(r"\s*-\s*", "-", text)

    return text

#updated parse fields to normalize data and ignore irrelevant letters that restrict
# methods ability to identify titles
def parse_fields(raw_text: str):
    result = {
        "name": None,
        "designation": None,
        "id_number": None,
        "issued_date": None,
        "expiry_date": None,
    }

    lines = [normalize_line(line) for line in raw_text.splitlines() if line.strip()]
    date_pattern = r"""
    \b(
        \d{1,2}\s*[-/]\s*[A-Za-z]{3,9}\s*[-/]\s*\d{4}     # 17-Nov-2020
        |
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4}                     # 04/05/2015
        |
        [A-Za-z]{3,9}\s*[-/]\s*\d{4}                      # Nov-2020
        |
        [A-Za-z]{3,9}\s+\d{4}                             # November 2020
    )\b
    """
    for line in lines:
        lower = line.lower()

        if result["name"] is None and ("name" in lower or "employee" in lower):
            m = re.search(r"(?:name|employee)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if m:
                value = clean_person_text(m.group(1))
                if value:
                    result["name"] = value

        if result["designation"] is None and ("designation" in lower or "desig" in lower):
            m = re.search(r"(?:designation|desig|id no)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if m:
                value = clean_person_text(m.group(1))
                if value:
                    result["designation"] = value

        ID_REGEX = r"\b(?=[A-Z0-9\-]{6,}\b)(?=.*\d)[A-Z0-9]+(?:[-–—][A-Z0-9]+)*\b"

        if result["id_number"] is None and "id" in lower:
            m = re.search(ID_REGEX, line, re.IGNORECASE)
            if m:
                result["id_number"] = re.sub(r"\s+", "", m.group())

        if result["issued_date"] is None and ("issued" in lower or "joined" in lower):
            m = re.search(date_pattern, line, re.IGNORECASE | re.VERBOSE)
            if m:
                result["issued_date"] = normalize_date(m.group())

        if result["expiry_date"] is None and ("expire" in lower or "expires" in lower):
            m = re.search(date_pattern, line, re.IGNORECASE | re.VERBOSE)
            if m:
                result["expiry_date"] = normalize_date(m.group())

    if result["id_number"] is None:
        for line in lines:
            m = re.search(r"\b\d{3,}(?:-\d+)+\b|\b\d{6,}\b", line)
            if m:
                result["id_number"] = m.group()
                break

    return result




@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"error": "Please upload a valid image file."}

        image_bytes = await file.read()
        image_bgr = load_image(image_bytes)

        if image_bgr is None:
            return {"error": "Image decoding failed"}

        rectified = image_bgr   # keep this for now
        # rectified = rectify_card(image_bgr)

        processed = preprocess_image(rectified)

        if processed is None:
            return {"error": "Preprocessing failed"}

        thresh = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        raw_text_gray = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
        raw_text_thresh = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")

        raw_text = raw_text_thresh if len(raw_text_thresh.strip()) > len(raw_text_gray.strip()) else raw_text_gray

        print("RAW GRAY:", repr(raw_text_gray))
        print("RAW THRESH:", repr(raw_text_thresh))
        print("CHOSEN RAW:", repr(raw_text))

        cv2.imwrite("debug_processed.jpg", processed)
        cv2.imwrite("debug_thresh.jpg", thresh)

        parsed = parse_fields(raw_text)

        return {
            "name": parsed["name"],
            "designation": parsed["designation"],
            "id_number": parsed["id_number"],
            "issued_date": parsed["issued_date"],
            "expiry_date": parsed["expiry_date"],
            "raw_text": raw_text,
        }

    except Exception as e:
        return {"error": str(e)}