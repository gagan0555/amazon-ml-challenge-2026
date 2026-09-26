import pandas as pd
from unidecode import unidecode
import re

def clean_and_romanize_series(series):
    """Cleans text, transliterates to ASCII, and removes corporate stop words."""
    def clean_text(text):
        if pd.isna(text): return ""
        text = str(text).strip()
        if not text: return ""
        
        text = unidecode(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        stop_words = [r'\binc\b', r'\bcorp\b', r'\bcorporation\b', r'\bpvt\b', 
                      r'\bprivate\b', r'\bltd\b', r'\blimited\b', r'\bllc\b', r'\bllp\b',
                      r'\bpraaivett\b', r'\blimittedd\b', r'\belelpii\b']
        
        for word in stop_words: 
            text = re.sub(word, ' ', text)
            
        return re.sub(r'\s+', ' ', text).strip()
        
    return series.apply(clean_text)