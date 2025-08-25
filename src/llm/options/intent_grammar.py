from __future__ import annotations
from typing import Optional, Dict, Any
import logging
from lark import Lark, Transformer, v_args
import re as _re
import hashlib
from .intent_models import Intent, Cardinality, Op, Obligations, LanceDBObligation

logger = logging.getLogger(__name__)

_GRAMMAR = r"""
start: query

query: FIND n_clause? LOCI WITH marker mid_clause? then_stage*
mid_clause: flank about? | about? flank | about

FIND: /find/i
// Accept both "loci" and "locus"
LOCI: /(loci|locus)/i
WITH: /with/i
THEN: /then/i
ON: /on/i
TO: /to/i

marker: MARKER_WORD (MARKER_WORD | "-" | "_" | "." )*
// A non-side-effecting marker term for use inside filters
marker_term: MARKER_WORD (MARKER_WORD | "-" | "_" | "." )*
MARKER_WORD: /[A-Za-z0-9]+/

// "five" | "5" | with comparators
n_clause: quant
// flank window: accepts both digits and number words via `number`
flank: "±" number
     | /flanking/i ("genes")? ("=")? number

quant: QOP? number
QOP: /exactly/i
   | /at\s+least/i
   | /at\s+most/i

about: /and\s+tell\s+me\s+about\s+them\.?/i

// stages: then LanceDB/embedding/kNN and literature
then_stage: THEN stage
stage: lancedb_stage | nn_stage | literature_stage

LDB: /lancedb|embedding|vector|knn/i
SEARCH: /search/i
LOOKUP: /lookup/i
BY: /by/i
COSINE: /cosine/i
SIMILARITY: /similarity/i
simphrase: BY COSINE SIMILARITY

// Support both orders: "closest 3", "3 closest"; allow synonyms (neighbors|relatives)
nn_spec: nn_after | nn_before
nn_after: (NEAREST|CLOSEST) NEIGH_WORD? EACH? number?
nn_before: number NEIGH_WORD? EACH? (NEAREST|CLOSEST)
NEAREST: /nearest/i
CLOSEST: /closest/i
NEIGH_WORD: /(neighbors?|relatives?)/i
EACH: /each/i

NOT: /not/i
ANNOTATED: /annotated/i
AS: /as/i
ldb_filter: NOT ANNOTATED AS marker_term BY ns
ns: PFAM | KOFAM
PFAM: /pfam/i
KOFAM: /kofam/i

literature_stage: /literature\s+search/i

// numbers: digits or words up to 20
number: INT | WORDNUM
WORDNUM: /(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)/i

PERFORM: /perform/i
A: /a/i

// LanceDB stage keywords + minimal optional connectors
lancedb_stage: PERFORM? A? LDB (SEARCH|LOOKUP)? (ON marker_term)? (nn_spec | number)? (TO marker_term)? simphrase? ldb_filter?

// Allow an NN-only stage without explicitly mentioning LanceDB
nn_stage: nn_spec simphrase? ldb_filter?

%import common.INT
%import common.WS
%ignore WS
%ignore /\*/
// Ignore common sentence punctuation
%ignore /[\.,;:!\?]/
// Ignore harmless filler words globally to reduce grammar complexity
%ignore /\b(?:a|an|the|of|for|me|them|those|identified|and|show|which|are|that|these|this|my|your|their)\b/i
"""

_WORDMAP = {
 "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
 "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20
}


@v_args(inline=True)
class _X(Transformer):
    def __init__(self, raw_text:str):
        self.raw_text = raw_text
        self.intent = Intent(marker="", raw_text=raw_text)

    # Collapse start -> query result
    def start(self, child):
        return child

    # marker
    def marker(self, *parts):
        m = "".join(str(p) for p in parts).strip()
        mlow = m.lower()
        # Minimal plural normalization for common markers
        if mlow == "integrases":
            mlow = "integrase"
        if mlow == "terminases":
            mlow = "terminase"
        self.intent.marker = mlow
        return m

    # quantifiers
    def quant(self, *items):
        # items: [QOP? , number]
        op = Op.EQ
        val = None
        if len(items) == 1:
            val = items[0]
        elif len(items) >= 2:
            q = str(items[0]).lower()
            if "at least" in q:
                op = Op.GE
            elif "at most" in q:
                op = Op.LE
            elif "exactly" in q:
                op = Op.EQ
            val = items[1]
        self.intent.N = Cardinality(value=int(val), op=op)
        return self.intent.N

    def number(self, tok):
        # tok: Token INT or WORDNUM
        if tok.type == 'INT':
            return int(tok.value)
        return _WORDMAP[tok.value.lower()]

    # flank: ±k or "flanking genes = k" (number already transformed)
    def flank(self, *items):
        # items contain one integer from `number`
        for it in items[::-1]:
            if isinstance(it, int):
                self.intent.flank = Cardinality(value=it, op=Op.EQ)
                return it
        return None

    # lancedb stage and filters
    def lancedb_stage(self, *items):
        self.intent.obligations.lancedb_knn.required = True
        # If a bare number followed LDB, treat as nn
        nn = None
        for it in items[::-1]:
            if isinstance(it, int):
                nn = it
                break
        if nn is not None:
            self.intent.nn = Cardinality(value=nn, op=Op.EQ)
            self.intent.obligations.lancedb_knn.nn = nn
        return self.intent

    def PFAM(self, *_): return "pfam"
    def KOFAM(self, *_): return "kofam"

    def ldb_filter(self, *items):
        # items sequence includes NOT, ANNOTATED, AS, marker_term, BY, ns
        ns = None
        mk = None
        for it in items:
            if isinstance(it, str):
                low = it.lower()
                if low in ("pfam", "kofam"):
                    ns = low
                else:
                    mk = low
        ldb = self.intent.obligations.lancedb_knn
        if ns:
            ldb.exclude_namespace = ns
        if mk:
            # simple singularization heuristic for common plural forms
            variants = [mk]
            if mk.endswith('es'):
                variants.append(mk[:-2])
            elif mk.endswith('s'):
                variants.append(mk[:-1])
            for v in variants:
                if v and v not in ldb.exclude_markers:
                    ldb.exclude_markers.append(v)
        try:
            logger.info(
                "GRAMMAR_LDB_FILTER: exclude_ns=%s exclude_markers=%s",
                ldb.exclude_namespace,
                ldb.exclude_markers,
            )
        except Exception:
            pass

    def marker_term(self, *parts):
        # Same as marker but without mutating self.intent.marker
        return "".join(str(p) for p in parts).strip()

    def nn_spec(self, *items):
        # trailing number optional
        nn = None
        for it in items[::-1]:
            if isinstance(it, int):
                nn = it
                break
        if nn is None:
            nn = 10  # deterministic default when nn mentioned but no number provided
        self.intent.nn = Cardinality(value=nn, op=Op.EQ)
        self.intent.obligations.lancedb_knn.required = True
        self.intent.obligations.lancedb_knn.nn = nn

    def literature_stage(self, *_):
        self.intent.obligations.literature = True

    def query(self, *_):
        # finalize defaults
        if self.intent.N.value is None:
            self.intent.N = Cardinality(value=5, op=Op.EQ)
        if self.intent.flank.value is None:
            self.intent.flank = Cardinality(value=4, op=Op.EQ)
        ldb = self.intent.obligations.lancedb_knn
        if ldb.required and (ldb.nn is None):
            ldb.nn = 10
            self.intent.nn = Cardinality(value=10, op=Op.EQ)
        return self.intent


def normalize_macro_text(text: str) -> str:
    """Public normalizer for macro intent text. Deterministic, no LLM.

    Applies tolerant rewrites so the strict grammar can parse natural phrasing.
    """
    def _normalize_freeform_phrases(t: str) -> str:
        nt = t
        try:
            # Map common synonyms
            nt = _re.sub(r"\bORFs?\b", "genes", nt, flags=_re.I)
            # Collapse common aside patterns into a flank spec (± N)
            nt = _re.sub(
                r"[,;:]?\s*(restricting|limiting)\s+to\b.*?\bwithin\s+(\d+)\s+genes?\b.*?(?=(?:[,;:.!?]|\bthen\b|$))",
                r" ± \2",
                nt,
                flags=_re.I | _re.S,
            )
            # If a clause like ", focusing on ... within N genes ..." exists, collapse it into a flank spec
            nt = _re.sub(
                r"[,;:]?\s*focusing\s+on\b.*?\bwithin\s+(\d+)\s+genes?\b.*?(?=(?:[,;:.!?]|\bthen\b|$))",
                r" ± \1",
                nt,
                flags=_re.I | _re.S,
            )
            # Generic mapping: "within N genes" → " ± N" (accepted by flank rule)
            nt = _re.sub(r"\bwithin\s+(\d+)\s+genes?\b", r" ± \1", nt, flags=_re.I)
            # Remove any residual adverbial clause like ", focusing on ..." up to punctuation or THEN
            nt = _re.sub(r"[,;:]?\s*focusing\s+on\b.*?(?=(?:[,;:.!?]|\bthen\b|$))", "", nt, flags=_re.I | _re.S)
            # Coarse fallback: strip trailing aside fragment starting with ", focusing" to end-of-sentence
            nt = _re.sub(r",\s*focusing\s+on[^.]*\.", ".", nt, flags=_re.I)
            # Whitespace tidy-up
            nt = _re.sub(r"\s{2,}", " ", nt)
            # Minimal plural normalization for marker mentions
            nt = _re.sub(r"\bintegrases\b", "integrase", nt, flags=_re.I)
            nt = _re.sub(r"\bterminases\b", "terminase", nt, flags=_re.I)
        except Exception:
            pass
        return nt

    return _normalize_freeform_phrases(text)


def parse_intent(text:str) -> Optional[Intent]:
    """Parse user text into a typed Intent. Logs detailed diagnostics at INFO level."""
    try:
        parser = Lark(_GRAMMAR, start="start", parser="lalr")
    except Exception as e:
        gh = hashlib.sha1(_GRAMMAR.encode("utf-8")).hexdigest()[:8]
        snippet = (text or "")[:160].replace("\n", " ")
        logger.info(f"GRAMMAR_COMPILE_FAIL: {e}")
        logger.info(f"GRAMMAR_SOURCE_HASH={gh} | ROUTER_TEXT='{snippet}'")
        return None
    try:
        tr = _X(text)
        norm = normalize_macro_text(text)
        if norm != text:
            snippet_before = (text or "")[:120].replace("\n", " ")
            snippet_after = (norm or "")[:120].replace("\n", " ")
            logger.info(f"GRAMMAR_NORMALIZED: before='{snippet_before}' | after='{snippet_after}'")
        intent: Intent = tr.transform(parser.parse(norm))
        # Post-parse deterministic capture for exclusion filter phrasing seen in wild text
        try:
            import re as _re
            if intent and intent.obligations and intent.obligations.lancedb_knn:
                ldb = intent.obligations.lancedb_knn
                if not ldb.exclude_namespace and not ldb.exclude_markers:
                    # tolerate emphasis markup around 'not' and common pluralization on marker
                    m = _re.search(r"\*?not\*?\s+annotated\s+as\s+([A-Za-z0-9._-]+)\s+by\s+(pfam|kofam)", text, _re.I)
                    if m:
                        mk = m.group(1).lower()
                        ns = m.group(2).lower()
                        ldb.exclude_namespace = ns
                        variants = [mk]
                        # Simple plural handling with special-case for 'integrases' -> 'integrase'
                        if mk == 'integrases':
                            variants.append('integrase')
                        if mk.endswith('es'):
                            variants.append(mk[:-2])
                        if mk.endswith('s') and mk[:-1] not in variants:
                            variants.append(mk[:-1])
                        for v in variants:
                            if v and v not in ldb.exclude_markers:
                                ldb.exclude_markers.append(v)
        except Exception:
            pass
        ldb = intent.obligations.lancedb_knn
        logger.info(
            "GRAMMAR_PARSE_OK: marker=%s N=%s flank=%s ldb_required=%s nn=%s exclude_ns=%s exclude_markers=%s",
            intent.marker,
            getattr(intent.N, "value", None),
            getattr(intent.flank, "value", None),
            getattr(ldb, "required", False),
            getattr(intent.nn, "value", None),
            getattr(ldb, "exclude_namespace", None),
            getattr(ldb, "exclude_markers", None),
        )
        return intent
    except Exception as e:
        snippet = (text or "")[:120].replace("\n", " ")
        logger.info(f"GRAMMAR_PARSE_FAIL: {e} | text='{snippet}…'")
        return None
