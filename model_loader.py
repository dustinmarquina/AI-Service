from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-small", device="mps" if __import__("torch").backends.mps.is_available() else "cpu")
    return _model