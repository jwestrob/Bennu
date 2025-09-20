from __future__ import annotations
import math

# Tunable constants (can be overridden via config injection in callers)
_ALPHA, _BETA, _GAMMA = 0.7, 0.3, 0.0
_LAMBDA = 1.0
_TAU = 0.1
_MIN_CONTIG_LEN = 1
_MIN_ORF_COUNT = 0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def evi_gate(seed: dict) -> bool:
    L = int(seed.get("contig_len") or 0)
    n_orf = int(seed.get("orf_count") or 0)
    ko_support = float(seed.get("ko_support") or 0.0)  # 0/1 if unavailable
    prior = 0.0  # placeholder for synteny prior if available

    if L < _MIN_CONTIG_LEN or n_orf < _MIN_ORF_COUNT:
        return False

    # Simple evidence score based on ORF density (no dom_score used)
    info = sigmoid(_BETA * (n_orf - _MIN_ORF_COUNT) + _GAMMA * prior + 0.0 * ko_support)
    cost = 1.0  # normalized unit cost for one neighborhood expansion
    evi = info - _LAMBDA * cost
    # Deterministic permissive gate: allow through by default
    return True
