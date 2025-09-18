from __future__ import annotations

"""
Finalizer utilities for Code Interpreter integration.

Builds analysis code programmatically based on available packs and the user's
visualization intent. Avoids prompt-specific hard-coding and uses only
matplotlib + pathlib + builtins for sandbox compatibility.
"""

from typing import Dict

def _wants(term: str, question: str) -> bool:
    q = (question or '').lower()
    return term in q

def build_ci_code(question: str, payload_path: str, plots_dir: str) -> str:
    """Return Python code for the interpreter based on question + payload path.

    The generated code:
    - Loads analysis_payload.json (via builtins.open)
    - Renders heatmaps for any matrices present (PFAM/KO top matrices)
    - Optionally renders bar charts for top features when requested
    - Saves figures into plots_dir

    Libraries: matplotlib (Agg), pandas, numpy, pathlib. No seaborn, no os.
    """
    want_heatmap = any(_wants(k, question) for k in ("heatmap", "visualize", "plot")) or True
    want_bars = any(_wants(k, question) for k in ("bar", "bars", "bar chart", "top 20", "top 10", "top10", "top"))

    # Template string (not an f-string). We replace placeholders explicitly below.
    code = """
import json
import builtins as bi
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

payload_path = r"__PAYLOAD_PATH__"
plots_dir = Path(r"__PLOTS_DIR__")
plots_dir.mkdir(parents=True, exist_ok=True)

# Load payload
payload = json.loads(bi.open(payload_path, 'r', encoding='utf-8').read())

def _to_df(matrix_rows, cols):
    if not matrix_rows or not cols:
        return pd.DataFrame()
    rows, idx = [], []
    for r in matrix_rows:
        idx.append(r.get('genome_id'))
        rows.append([int(r.get(c, 0)) for c in cols])
    return pd.DataFrame(rows, index=idx, columns=cols)

def _save_heatmap(df, title, name, _plots_dir=plots_dir, normalize=True, log1p=True):
    if df is None or df.empty:
        return None
    import numpy as _np
    X = df.values.astype(float)
    if log1p:
        X = _np.log1p(X)
    if normalize:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, ddof=0, keepdims=True) + 1e-9
        Z = (X - mu) / sigma
        data = Z
        cmap = 'RdBu_r'
        vmin, vmax = -2.0, 2.0
        cbar_label = 'z-score (log1p counts)'
    else:
        data = X
        cmap = 'viridis'
        vmin = vmax = None
        cbar_label = 'count'
    plt.figure(figsize=(max(7, df.shape[1]*0.6), max(3.5, df.shape[0]*0.6)))
    im = plt.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    try:
        cbar.set_label(cbar_label, fontsize=9)
    except Exception:
        pass
    plt.xticks(range(len(df.columns)), list(df.columns), rotation=60, ha='right', fontsize=9)
    plt.yticks(range(len(df.index)), list(df.index), fontsize=9)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    out = _plots_dir / name
    plt.savefig(str(out), dpi=150)
    plt.close()
    return str(out)

def _save_bars(items, title, name, _plots_dir=plots_dir, limit=10):
    if not items:
        return None
    # Sort by count desc then ID/name and take top-N
    items2 = sorted(items, key=lambda x: (-(int(x.get('count') or x.get('total') or 0)), str(x.get('id') or x.get('name') or '')))
    items2 = items2[:max(1, int(limit))]
    labels = [str(x.get('name') or x.get('id')) for x in items2]
    counts = [int(x.get('count') or x.get('total') or 0) for x in items2]
    if not any(counts):
        return None
    plt.figure(figsize=(max(7, len(labels)*0.5), 4.5))
    plt.bar(range(len(labels)), counts, color='#4c78a8')
    plt.xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=9)
    plt.ylabel('count', fontsize=10)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    out = _plots_dir / name
    plt.savefig(str(out), dpi=150)
    plt.close()
    return str(out)

outs = []

# Matrices → heatmaps
try:
    m = (payload.get('feature_profile') or {}).get('per_genome_top_matrix') or {}
    order = (m.get('feature_order') or {})
    if __WANT_HEATMAP__:
        # PFAM
        pf_rows = m.get('pfam') or []
        pf_cols = order.get('pfam') or []
        print(f"PFAM rows={len(pf_rows)} cols={len(pf_cols)}")
        pf_df = _to_df(pf_rows, pf_cols)
        try:
            print(f"PFAM df shape={pf_df.shape}")
        except Exception:
            pass
        pf_png = _save_heatmap(pf_df, 'PFAM feature heatmap (z-score by feature)', 'feature_heatmap_pfam.png')
        if pf_png: outs.append(pf_png)
        # KO
        ko_rows = m.get('ko') or []
        ko_cols = order.get('ko') or []
        print(f"KO rows={len(ko_rows)} cols={len(ko_cols)}")
        ko_df = _to_df(ko_rows, ko_cols)
        try:
            print(f"KO df shape={ko_df.shape}")
        except Exception:
            pass
        ko_png = _save_heatmap(ko_df, 'KO feature heatmap (z-score by feature)', 'feature_heatmap_ko.png')
        if ko_png: outs.append(ko_png)
except Exception as e:
    import traceback as _tb
    print('ERROR: heatmap rendering failed:', e)
    _tb.print_exc()

# FeatureProfileSummary → bar charts
try:
    if __WANT_BARS__:
        summ = (payload.get('feature_profile') or {}).get('summary') or {}
        top_pfam = summ.get('top_pfam') or []
        top_ko = summ.get('top_ko') or []
        print(f"Bars: top_pfam={len(top_pfam)} top_ko={len(top_ko)}")
        pf_bar = _save_bars(top_pfam, 'Top PFAM features (global totals)', 'bars_top_pfam.png', limit=10)
        if pf_bar: outs.append(pf_bar)
        ko_bar = _save_bars(top_ko, 'Top KO features (global totals)', 'bars_top_ko.png', limit=10)
        if ko_bar: outs.append(ko_bar)
except Exception as e:
    import traceback as _tb
    print('ERROR: bar chart rendering failed:', e)
    _tb.print_exc()

print('Generated figures:')
for o in outs:
    print(o)
try:
    from pathlib import Path as _P
    print('CWD:', _P.cwd())
    print('PNG files in CWD:')
    for f in _P('.').glob('*.png'):
        print(str(f))
except Exception:
    pass
"""
    # Replace placeholders
    code = code.replace('__PAYLOAD_PATH__', payload_path)
    code = code.replace('__PLOTS_DIR__', plots_dir)
    code = code.replace('__WANT_HEATMAP__', 'True' if want_heatmap else 'False')
    code = code.replace('__WANT_BARS__', 'True' if want_bars else 'False')
    return code
