from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pytesseract
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#testing pytesseract
@app.get("/test-tesseract")
def test_tesseract():
    import subprocess
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[:200],
            "stderr": result.stderr[:200],
        }
    except Exception as e:
        return {"error": str(e)}
    
# before preprocessing, need to rectify card first by
# 1. finding the boundary of the card vs the surface its on
# 2. warp card so it becomes 2D (removes any angling from the way the pic is took)
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


def find_card_contour(edges, image_shape):
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        # ignore tiny contours
        if area < 0.10 * img_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)

        # id cards are wider than tall
        if ratio < 1.2 or ratio > 2.2:
            continue

        # reject contours that are mostly only in the very top region
        if y + h < img_h * 0.35:
            continue

        return approx

    return None


def rectify_card(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    card_contour = find_card_contour(edges, image_bgr.shape)

    if card_contour is None:
        print("No valid card contour found, using original image")
        return image_bgr

    warped = four_point_transform(image_bgr, card_contour)

    if warped is None or warped.size == 0:
        print("Warp failed, using original image")
        return image_bgr

    h, w = warped.shape[:2]
    ratio = w / float(h)

    if w < 300 or h < 150:
        print("Warp too small, using original image")
        return image_bgr

    if ratio < 1.2 or ratio > 2.2:
        print("Warp ratio suspicious, using original image")
        return image_bgr

    return warped

# since my ID parser searches for a pattern of numbers, sometimes dates
# get considered as ID number if no ID is found. To avoid this, 
# this function looks for the regex of the date formats that I'm
# looking for and avoids those 
def looks_like_date(text: str) -> bool:
    date_like_pattern = r"""
    ^\s*(
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4}
        |
        \d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+\d{4}
        |
        [A-Za-z]{3,9}\s+\d{4}
        |
        [A-Za-z]{3,9}[/-]\d{4}
    )\s*$
    """
    return re.match(date_like_pattern, text, re.IGNORECASE | re.VERBOSE) is not None

# preprpcessing image so ocr can understand text better
def preprocess_image(image_bgr):
    #resizing img (3x) so ocr can read it better
    enlarged = cv2.resize(
        image_bgr,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    #converting from color to grayscale so ocr can clearly see the 
    #intensity difference between text and background
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    #using histogram equalization to enhance contrast within regions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    #using gaussian blur to remove high f. noise to smooth out grainy
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    #high contrast grayscale imahe is now ready for the OCR
    return gray


@app.get("/")
def root():
    return {"message": "Backend is up & running!"}


# normalizing ID line since theres so many variations of its title
def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    line = line.replace("IDNo", "ID No")
    line = line.replace("IDNO", "ID No")
    line = line.replace("IdNo", "ID No")
    line = line.replace("EmployeeNumber", "Employee Number")
    line = line.replace("EmployeeNo", "Employee No")
    return line


def clean_person_text(text: str):
    text = text.strip()

    # remove common separators like / : 
    text = text.replace("|", " ")
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # use regex to keep splits that are similar to a word format
    words = text.split()
    words = [w for w in words if re.match(r"^[A-Za-z\-\.]+$", w)]

    if not words:
        return None

    # ideally, there's two maybe 3 capital letters when identifying name
    best = None
    best_score = -1

    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words) + 1)):
            candidate_words = words[i:j]
            candidate = " ".join(candidate_words)

            # ignore text common text that my processor often confuses as peoples names
            if candidate.lower() in ["company", "employee", "card"]:
                continue

            # set a score where phrases with caps are rankes higher 
            if re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}$", candidate):
                score = len(candidate_words) * 10 + len(candidate)
                if score > best_score:
                    best = candidate
                    best_score = score

    if best:
        return best

    # fallback: if nothing is found, result to longest phrase with letters only
    cleaned = " ".join(words)
    if len(cleaned) >= 4 and re.search(r"[A-Za-z]", cleaned):
        return cleaned

    return None


def clean_id_text(text: str):
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"[^A-Za-z0-9/\-]+$", "", text)

    if len(text) < 4:
        return None
    if not re.search(r"\d", text):
        return None
    if looks_like_date(text):
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


# helper for dates on separate row from their titles
def extract_dates_from_line(line: str):
    date_pattern = r"""
    \b(
        \d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+\d{4}
        |
        \d{1,2}\s*[-/]\s*[A-Za-z]{3,9}\s*[-/]\s*\d{4}
        |
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4}
        |
        [A-Za-z]{3,9}\s*[-/]\s*\d{4}
        |
        [A-Za-z]{3,9}\s+\d{4}
    )\b
    """
    return re.findall(date_pattern, line, re.IGNORECASE | re.VERBOSE)


# updated parse fields to normalize data and ignore irrelevant letters that restrict
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
    ID_REGEX = r"\b(?=[A-Z0-9/\-]{5,}\b)(?=.*\d)[A-Z0-9]+(?:[/-][A-Z0-9]+)*\b"

    for i, line in enumerate(lines):
        lower = line.lower()

        # if dates are on separate line from the header, identify them

        if result["name"] is None and ("name" in lower or "employee" in lower):
            m = re.search(r"(?:name|employee)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if m:
                value = clean_person_text(m.group(1))
                if value and value.upper() != "ID CARD":
                    result["name"] = value
            elif i + 1 < len(lines):
                next_value = clean_person_text(lines[i + 1])
                if next_value:
                    result["name"] = next_value

        if result["designation"] is None and (
            "designation" in lower or "desig" in lower or "position" in lower
        ):
            m = re.search(r"(?:designation|desig|position)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
            if m:
                value = clean_person_text(m.group(1))
                if value:
                    result["designation"] = value
            elif i + 1 < len(lines):
                next_value = clean_person_text(lines[i + 1])
                if next_value:
                    result["designation"] = next_value

        if result["id_number"] is None and (
            "id" in lower or "employee number" in lower or "employee no" in lower
        ):
            m = re.search(
                r"(?:id\s*no|id|employee\s*number|employee\s*no)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\- ]{3,})",
                line,
                re.IGNORECASE
            )
            if m:
                candidate = clean_id_text(m.group(1))
                if candidate:
                    result["id_number"] = candidate
            else:
                m = re.search(ID_REGEX, line, re.IGNORECASE)
                if m:
                    candidate = clean_id_text(m.group())
                    if candidate:
                        result["id_number"] = candidate

        #instant fallback for id, locating 
        if result["id_number"] is None and ("id" in lower or "no" in lower):
            matches = re.findall(r"\b\d{6,}\b", line)
            if matches:
                result["id_number"] = max(matches, key=len)

        # in the cases where dates (issued and expiry) and their headers are on different lines
        if ("from" in lower or "issued" in lower or "joined" in lower) and ("expire" in lower):
            dates = extract_dates_from_line(line)
            if len(dates) < 2 and i + 1 < len(lines):
                dates += extract_dates_from_line(lines[i + 1])

            if len(dates) >= 2:
                if result["issued_date"] is None:
                    result["issued_date"] = normalize_date(dates[0])
                if result["expiry_date"] is None:
                    result["expiry_date"] = normalize_date(dates[1])
                continue
        #just for issued date (on same line as header)
        if result["issued_date"] is None and ("issued" in lower or "joined" in lower or "from" in lower or "iss" in lower):
            dates = extract_dates_from_line(line)
            if not dates and i + 1 < len(lines):
                dates = extract_dates_from_line(lines[i + 1])
            if dates:
                result["issued_date"] = normalize_date(dates[0])
        #just for expiry date (on same line as header)
        if result["expiry_date"] is None and ("expire" in lower or "expires" in lower or "expiry" in lower or "exp" in lower):
            dates = extract_dates_from_line(line)
            if not dates and i + 1 < len(lines):
                dates = extract_dates_from_line(lines[i + 1])
            if dates:
                result["expiry_date"] = normalize_date(dates[0])

    # fallback for id if it wasn't found originally 
    if result["id_number"] is None:
        for line in lines:
            lower = line.lower()

            if "id" not in lower and "employee number" not in lower and "employee no" not in lower:
                continue

            m = re.search(ID_REGEX, line, re.IGNORECASE)
            if m:
                candidate = clean_id_text(m.group())
                if candidate:
                    result["id_number"] = candidate
                    break

    return result


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        #ensure file type is correct for image
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"error": "Please upload a valid image file."}

        #read img byte and convert to openCV (BGR) format
        image_bytes = await file.read()
        image_bgr = load_image(image_bytes)
        
        #ensure image was actually decoded
        if image_bgr is None:
            return {"error": "Image decoding failed"}

        # try OCR on decoded image
        processed_original = preprocess_image(image_bgr)

        if processed_original is None:
            return {"error": "Preprocessing failed"}

        #applying adaptive threshilfing to increase text contrast
        thresh_original = cv2.adaptiveThreshold(
            processed_original,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        #apply ocr to grayscale and thresh.
        raw_text_gray_original = pytesseract.image_to_string(
            processed_original,
            config="--oem 3 --psm 6"
        )
        raw_text_thresh_original = pytesseract.image_to_string(
            thresh_original,
            config="--oem 3 --psm 6"
        )

        # choose the better ocr result based on length of text
        best_original = (
            raw_text_thresh_original
            if len(raw_text_thresh_original.strip()) > len(raw_text_gray_original.strip())
            else raw_text_gray_original
        )

        # try rectified image too, but only keep it if it is actually better
        rectified = rectify_card(image_bgr)
        processed_rectified = preprocess_image(rectified)

        raw_text_rectified = ""
        if processed_rectified is not None:
            thresh_rectified = cv2.adaptiveThreshold(
                processed_rectified,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15
            )

            raw_text_gray_rectified = pytesseract.image_to_string(
                processed_rectified,
                config="--oem 3 --psm 6"
            )
            raw_text_thresh_rectified = pytesseract.image_to_string(
                thresh_rectified,
                config="--oem 3 --psm 6"
            )

            raw_text_rectified = (
                raw_text_thresh_rectified
                if len(raw_text_thresh_rectified.strip()) > len(raw_text_gray_rectified.strip())
                else raw_text_gray_rectified
            )

            cv2.imwrite("debug_rectified.jpg", rectified)
            cv2.imwrite("debug_processed_rectified.jpg", processed_rectified)
            cv2.imwrite("debug_thresh_rectified.jpg", thresh_rectified)

        # choose whichever OCR result is better
        raw_text = best_original
        chosen_processed = processed_original
        chosen_thresh = thresh_original

        if len(raw_text_rectified.strip()) > len(best_original.strip()):
            raw_text = raw_text_rectified
            chosen_processed = processed_rectified
            chosen_thresh = thresh_rectified

        print("RAW ORIGINAL:", repr(best_original))
        print("RAW RECTIFIED:", repr(raw_text_rectified))
        print("CHOSEN RAW:", repr(raw_text))

        cv2.imwrite("debug_processed.jpg", chosen_processed)
        cv2.imwrite("debug_thresh.jpg", chosen_thresh)

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