from pathlib import Path
import csv
import gc
import cv2

from rotate_serial import crop_by_text_contour, model
from ocr import read_serial, imread_unicode
from postprocess import extract_serial

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / 'output' / 'test_images'
OUTPUT_CSV = ROOT / 'output' / 'results.csv'

DEBUG_DIR = ROOT / 'output' / 'debug_crops'
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SERIAL_LEN = 11

def save_crop(path, img):
    ok, buf = cv2.imencode('.png', img)
    if ok:
        buf.tofile(str(path))

def _box_priority(crop):
    h, w = crop.shape[:2]
    is_vertical = h > w * 1.2
    brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
    return (1 if is_vertical else 0, brightness)


def _length_penalty(fixed):
    if not fixed:
        return 999
    return abs(len(fixed) - EXPECTED_SERIAL_LEN)

def process_image(img_path):
    img = imread_unicode(str(img_path))
    if img is None:
        print(f'{img_path.name}: не удалось открыть файл')
        return None

    for f in DEBUG_DIR.glob(f'{img_path.stem}_*'):
        f.unlink()


    result = model.predict(img, conf=0.1, verbose=False)[0]

    nominal = None
    number_boxes = [] 
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        conf = float(box.conf)

        if class_name.startswith('rub-') and nominal is None:
            nominal = class_name.split('-')[-1]
        elif class_name == 'number' and conf > 0.2: 
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            number_boxes.append({'crop': crop,
                                 'priority':_box_priority(crop)
                                 })

    if not number_boxes:
        print(f'{img_path.name}: серийный номер не найден')
        del img, result
        gc.collect()
        return None

    number_boxes.sort(key=lambda b: b['priority'], reverse=True)

    best_result = None
    best_key = None
    best_raw = ''

    for idx, b in enumerate(number_boxes):
        aligned = crop_by_text_contour(b['crop'])
        if aligned is None:
            aligned = b['crop']
            name = f'{img_path.stem}_num{idx}'

        name = f'{img_path.stem}_num{idx}'   
        save_crop(DEBUG_DIR / f'{name}_aligned.png', aligned)


        raw_text = read_serial(aligned)
        parsed = extract_serial(raw_text)


        if parsed['valid']:
            best_result = parsed
            best_raw = raw_text
            break


        v, br = b['priority']
        key = (_length_penalty(parsed['fixed']), -v, -br)
        if best_key is None or key < best_key:
            best_key = key
            best_result = parsed
            best_raw = raw_text

    if best_result is None:
        print(f'{img_path.name}: ни один кроп не обработан')
        del img, result
        del number_boxes
        gc.collect()
        return None


    final_serial = best_result['fixed'] or ''
    print(f"{img_path.name:15} | {nominal or '?':5} | {final_serial or 'не распознано'}")

    del img, result
    for b in number_boxes:
        del b['crop']
    del number_boxes
    gc.collect()

    return {
        'file': img_path.name,
        'nominal': nominal or '',
        'raw_ocr': best_raw,
        'serial': best_result['fixed'] or '',
        'valid': best_result['valid']
    }


if __name__ == '__main__':
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for img_path in sorted(FOLDER.glob('*')):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        res = process_image(img_path)
        if res:
            rows.append(res)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(
            f, fieldnames=['file', 'nominal', 'raw_ocr', 'serial', 'valid']
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nГотово! Обработано файлов: {len(rows)}")