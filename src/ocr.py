# распознавание серийного номера купюры
import cv2
import easyocr
import numpy as np 

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['ru'], gpu=False)
    return _reader


ALLOWLIST = (
    'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    'абвгдежзийклмнопрстуфхцчшщъыьэюя'
    '0123456789'
)


def _preprocess_crop(crop, scale=2):
    
    if crop is None or crop.size == 0:
        print("Пустой кроп - нечего обрабатывать")
        return None

    h, w = crop.shape[:2]
    crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return binary


def read_serial(crop, scale=4):

    processed = _preprocess_crop(crop, scale=scale)
    
    if processed is None:
        return ''
    
    reader = _get_reader()
    results = reader.readtext(processed, detail=0, paragraph=False, 
                              allowlist=ALLOWLIST)

    text = ''.join(results)
    text = text.replace(' ', '')
    return text
       
        
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return img


if __name__ == '__main__':
    import sys 

    if len(sys.argv) < 2:
        print('Использование: python ocr.py <путь к изображению>')
        sys.exit(1)

    image_path = sys.argv[1]
    img = imread_unicode(image_path)

    if img is None:
        print(f"Не удалось открыть: {image_path}")
        sys.exit(1)

    text = read_serial(img)

    print(f"Распознано: {text!r}")

   
    
