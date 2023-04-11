import os
from sentence_transformers import SentenceTransformer

import numpy as np

from typing import List, Dict, Union

def load_model(model_str: str = "all-mpnet-base-v2") -> SentenceTransformer:
    return SentenceTransformer(model_str)

def encode_sentences(model: SentenceTransformer, sentences: Union[str, List[str]]):
    return model.encode(sentences)