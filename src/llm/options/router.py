from __future__ import annotations
import re
import os
from typing import Optional
from .intent_grammar import parse_intent, normalize_macro_text
from .intent_models import Intent, Cardinality, Op

USE_GRAMMAR_ROUTER = os.getenv("USE_GRAMMAR_ROUTER", "1") not in ("0", "false", "False")


def _regex_fallback(user_text: str) -> Optional[Intent]:
    _PATTERN_MARKER = r"(integrase|terminase|recombinase|transposase|cas\d+|recA|ligase|pol[A-Z0-9]+)"
    _PATTERN_FIND_LOCI = re.compile(
        r"find\s+(?P<N>\d+)\s+loci\s+with\s+(?P<marker>%s)" % _PATTERN_MARKER, re.I
    )
    _PATTERN_FLANK = re.compile(r"(?:flank(?:ing)?\s*=\s*|±|plus\/minus|\+\/-)\s*(?P<k>\d+)", re.I)
    _PATTERN_NN_A = re.compile(r"(?:closest|nearest)\s*(?P<nn>\d+)", re.I)
    _PATTERN_NN_B = re.compile(r"(?P<nn>\d+)\s*(?:closest|nearest)", re.I)

    m = _PATTERN_FIND_LOCI.search(user_text)
    if m:
        N = int(m.group("N"))
        marker = m.group("marker").lower()
        k = 4
        nn = None
        m2 = _PATTERN_FLANK.search(user_text)
        if m2:
            k = int(m2.group("k"))
        m3 = _PATTERN_NN_A.search(user_text) or _PATTERN_NN_B.search(user_text)
        if m3:
            nn = int(m3.group("nn"))

        intent = Intent(marker=marker, raw_text=user_text)
        intent.N = Cardinality(value=N, op=Op.EQ)
        intent.flank = Cardinality(value=k, op=Op.EQ)
        if nn is not None:
            intent.nn = Cardinality(value=nn, op=Op.EQ)
            intent.obligations.lancedb_knn.required = True
            intent.obligations.lancedb_knn.nn = nn
        return intent

    knn_pattern = re.search(r"(?:nearest|closest)\s+(\d+)\s+(?:proteins|neighbors?)", user_text, re.I)
    if knn_pattern:
        nn = int(knn_pattern.group(1))
        intent = Intent(marker="", raw_text=user_text)
        intent.nn = Cardinality(value=nn, op=Op.EQ)
        intent.obligations.lancedb_knn.required = True
        intent.obligations.lancedb_knn.nn = nn
        return intent

    generic_knn = re.search(r"(closest|nearest)\s+(?:proteins|neighbors?)", user_text, re.I)
    if generic_knn:
        intent = Intent(marker="", raw_text=user_text)
        intent.nn = Cardinality(value=10, op=Op.EQ)
        intent.obligations.lancedb_knn.required = True
        intent.obligations.lancedb_knn.nn = 10
        return intent

    return None


def parse_macro_intent(user_text: str) -> Optional[Intent]:
    if USE_GRAMMAR_ROUTER:
        try:
            norm = normalize_macro_text(user_text)
            if norm != user_text:
                before = (user_text or "")[:120].replace("\n", " ")
                after = (norm or "")[:120].replace("\n", " ")
                # Mirror grammar module logging key to make it easy to grep
                print(f"INFO:GRAMMAR_NORMALIZED(router): before='{before}' | after='{after}'")
            return parse_intent(norm)
        except Exception:
            return parse_intent(user_text)
    return _regex_fallback(user_text)
