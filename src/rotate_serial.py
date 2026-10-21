import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'best.pt'
model = YOLO(str(MODEL_PATH))

def _best_number_box(r):
    best_conf, best_box = 0.0, None
    for box in r.boxes:
        if model.names[int(box.cls)] != 'number':
            continue
        conf = float(box.conf)
        if conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0].tolist()
    return best_box


def find_best_serial(img, result=None):
    best_box = _best_number_box(result)
    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def crop_by_text_contour(crop, scale=2, padding=5):
    if crop is None or crop.size == 0:
        return None

    h_crop, w_crop = crop.shape[:2]
    if h_crop > w_crop * 1.3:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        h_crop, w_crop = crop.shape[:2]

    h, w = crop.shape[:2]
    big = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return crop

    pts = np.vstack([c.reshape(-1, 2) for c in contours])
    
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    d01 = np.linalg.norm(box[0] - box[1])
    d12 = np.linalg.norm(box[1] - box[2])

    if d01 > d12:
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]

    else:
        dx = box[2][0] - box[1][0]
        dy = box[2][1] - box[1][1]

    angle = np.degrees(np.arctan2(dy, dx))

    if angle > 90:
        angle -= 180

    elif angle < -90:
        angle += 180

    if abs(angle) < 5:
        return crop

    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos)) + 2 * padding
    nH = int((h * cos) + (w * sin)) + 2 * padding

    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    rotated = cv2.warpAffine(
        crop, M, (nW, nH), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )


    return rotated
