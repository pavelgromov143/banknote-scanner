import re 


SERIAL_PATTERN = re.compile(r'^[А-Яа-я]{2}\d{7,9}$')

LAT_TO_CYR = str.maketrans({
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'Y': 'У'
})

DIGIT_FIX = str.maketrans({
    'O': '0', 'О': '0',
    'I': '1', 'l': '1',
    'S': '5', 'З': '3', 'з': '3'
})

def _clean_serial(text):
    text = text or ''
    text = text.translate(LAT_TO_CYR)
    return re.sub(r'[^А-Яа-яA-za-z\d]', '', text )


def extract_serial(raw_text):
    cleaned = _clean_serial(raw_text)
    
    letters = "".join(c for c in cleaned if c.isalpha())[:2]
    digits = "".join(c for c in cleaned if c.isdigit())
    digits = digits.translate(DIGIT_FIX)
    digits = re.sub(r'\D', '', digits)

    fixed = letters + digits if len(letters) == 2 else None
    valid = bool(fixed and SERIAL_PATTERN.fullmatch(fixed))

    return {
        'raw': raw_text,
        'fixed': fixed,
        'valid': valid
    }