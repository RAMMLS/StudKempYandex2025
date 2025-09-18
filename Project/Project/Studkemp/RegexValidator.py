import re

#print(COMPILED_PATTERNS)
def detect_injection(text: str, COMPILED_PATTERNS) -> bool:
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return True
    return False

def get_detected_pattern(text: str, COMPILED_PATTERNS) -> str:
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return ""