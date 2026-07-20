# TODO
from __future__ import annotations
from typing import List
import re

from sudachipy import dictionary, tokenizer

_TOKENIZER = dictionary.Dictionary().create()
_SPLIT_MODE = tokenizer.Tokenizer.SplitMode.C

# particle settings (removing)
_JA_STOPWORDS = {
    "の","に","は","を","が","へ","と","で","や","も",
    "ね","よ","な","だ","です","ます",
}

# Remove symbols, keep single Japanese/Kanji characters
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_ja_for_bm25(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    ms = _TOKENIZER.tokenize(text, _SPLIT_MODE)
    tokens: List[str] = []
    for m in ms:
        tok = m.normalized_form().strip()
        if not tok:
            continue
        if tok in _JA_STOPWORDS:
            continue
        # symbols removing (예: "、", "。", "・", "（" )
        if _PUNCT_RE.match(tok):
            continue
        tokens.append(tok)
    return tokens
