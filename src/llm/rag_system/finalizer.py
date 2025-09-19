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
import hashlib
import textwrap as _tw

payload_path = r"__PAYLOAD_PATH__"
plots_dir = Path(r"__PLOTS_DIR__")
plots_dir.mkdir(parents=True, exist_ok=True)

# Load payload
payload = json.loads(bi.open(payload_path, 'r', encoding='utf-8').read())
# Label maps (e.g., pfam accession -> short name)
_labels_map = ((payload.get('feature_profile') or {}).get('summary') or {}).get('labels') or {}
def _name_for(ft, fid):
    try:
        fmap = _labels_map.get(ft) or {}
        s = str(fid)
        return fmap.get(s) or fmap.get(s.upper()) or fmap.get(s.lower()) or s
    except Exception:
        return str(fid)
question_text = str((payload or {}).get('question') or '')
def _tag(name: str) -> str:
    h = hashlib.md5((question_text + '|' + str(name)).encode('utf-8')).hexdigest()
    return h[:8]

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
        # placeholder figure with message
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No signal to plot\\n" + str(title), ha='center', va='center')
        plt.axis('off')
        out = _plots_dir / name
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(out)
    import numpy as _np
    X = df.values.astype(float)
    if log1p:
        X = _np.log1p(X)
    if not _np.any(X):
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "All-zero matrix\\n" + str(title), ha='center', va='center')
        plt.axis('off')
        out = _plots_dir / name
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(out)
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
    # Square canvas with cell-based sizing and label allowances to avoid scrunching
    nrows, ncols = int(df.shape[0]), int(df.shape[1])
    max_xlab = max([len(str(c)) for c in df.columns] + [6])
    max_ylab = max([len(str(r)) for r in df.index] + [6])
    cell = 0.6  # inches per cell side
    w_cells = max(3.5, ncols * cell)
    h_cells = max(3.5, nrows * cell)
    w_allow = max_xlab * 0.18
    h_allow = max_ylab * 0.14
    side = max(7.0, w_cells, h_cells, w_allow, h_allow)
    plt.figure(figsize=(side, side))
    im = plt.imshow(data, aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
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
    plt.savefig(str(out), dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return str(out)

def _abbrev_label(s, maxlen=16):
    s = str(s)
    return s if len(s) <= maxlen else (s[:maxlen-1] + '…')

def _save_bars(items, title, name, _plots_dir=plots_dir, limit=10, use_names=False):
    if not items:
        return None
    # Sort by count desc then ID/name and take top-N
    items2 = sorted(items, key=lambda x: (-(int(x.get('count') or x.get('total') or 0)), str(x.get('id') or x.get('name') or '')))
    items2 = items2[:max(1, int(limit))]
    ids = [str(x.get('id')) for x in items2]
    names = [str(x.get('name') or x.get('id')) for x in items2]
    # Prefer concise IDs in the plot; write a legend mapping file alongside
    labels = [_abbrev_label(n if use_names else i, 12) for i, n in zip(ids, names)]
    counts = [int(x.get('count') or x.get('total') or 0) for x in items2]
    if not any(counts):
        return None
    # Scale width by count and label length
    max_lab = max([len(l) for l in labels] + [6])
    plt.figure(figsize=(max(7, len(labels)*0.7, max_lab*0.4), 5.0))
    plt.bar(range(len(labels)), counts, color='#4c78a8')
    plt.xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=9)
    plt.ylabel('count', fontsize=10)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    out = _plots_dir / name
    plt.savefig(str(out), dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # Write legend mapping
    try:
        legend = _plots_dir / (name.replace('.png','') + '_legend.txt')
        with open(legend, 'w', encoding='utf-8') as lf:
            for i, n in zip(ids, names):
                lf.write(str(i))
                lf.write('\\t')
                lf.write(str(n))
                lf.write('\\n')
    except Exception:
        pass
    return str(out)

outs = []

# Matrices → heatmaps
try:
    m = (payload.get('feature_profile') or {}).get('per_genome_top_matrix') or {}
    order = (m.get('feature_order') or {})
    if __WANT_HEATMAP__:
        ftypes = list((order or {}).keys())
        for ft in ftypes:
            rows = m.get(ft) or []
            cols = order.get(ft) or []
            print(f"{ft.upper()} rows={len(rows)} cols={len(cols)}")
            df = _to_df(rows, cols)
            try:
                print(f"{ft.upper()} df shape={df.shape}")
            except Exception:
                pass
            tag = _tag(ft)
            # Prefer PFAM short IDs (names) instead of accessions on heatmap axes
            if ft == 'pfam' and not df.empty:
                try:
                    disp_cols = [ _name_for('pfam', c) for c in list(df.columns) ]
                    _df_disp = df.copy()
                    _df_disp.columns = disp_cols
                except Exception:
                    _df_disp = df
            else:
                _df_disp = df
            # PCA-based ordering for readability
            def _pca_order(_A, axis=0):
                import numpy as _np
                A = _A.astype(float)
                if axis == 0:  # order rows (genomes)
                    X = A - A.mean(axis=0, keepdims=True)
                    try:
                        U, S, Vt = _np.linalg.svd(X, full_matrices=False)
                        order = _np.argsort(U[:,0])
                    except Exception:
                        order = _np.arange(A.shape[0])
                    return order
                else:  # order columns (features)
                    X = A - A.mean(axis=1, keepdims=True)
                    try:
                        U, S, Vt = _np.linalg.svd(X.T, full_matrices=False)
                        order = _np.argsort(U[:,0])
                    except Exception:
                        order = _np.arange(A.shape[1])
                    return order
            try:
                if not _df_disp.empty and _df_disp.shape[0] > 1 and _df_disp.shape[1] > 1:
                    import numpy as _np
                    rord = _pca_order(_df_disp.values, axis=0)
                    cord = _pca_order(_df_disp.values, axis=1)
                    _df_disp = _df_disp.iloc[list(rord), :]
                    _df_disp = _df_disp.iloc[:, list(cord)]
            except Exception:
                pass
            fname = f"feature_heatmap_{ft}_{tag}.png"
            title = f"{ft.upper()} feature heatmap (z-score by feature)"
            png = _save_heatmap(_df_disp, title, fname)
            if png: outs.append(png)
            # Genome correlation heatmap (only when question suggests similarity/clustering)
            try:
                ql = (question_text or '').lower()
                want_corr = any(k in ql for k in ('cluster','similarity','correlation','compare'))
                if want_corr and (not _df_disp.empty) and _df_disp.shape[0] > 1 and _df_disp.shape[1] > 1:
                    import numpy as _np
                    X = _df_disp.values.astype(float)
                    C = _np.corrcoef(X)
                    plt.figure(figsize=(max(6, _df_disp.shape[0]*0.6), max(5, _df_disp.shape[0]*0.6)))
                    im = plt.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
                    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                    try:
                        cbar.set_label('Pearson r', fontsize=9)
                    except Exception:
                        pass
                    plt.xticks(range(len(_df_disp.index)), list(_df_disp.index), rotation=60, ha='right', fontsize=9)
                    plt.yticks(range(len(_df_disp.index)), list(_df_disp.index), fontsize=9)
                    plt.title(f"Genome correlation ({ft.upper()})", fontsize=11)
                    plt.tight_layout()
                    corr_name = f"genome_corr_{ft}_{tag}.png"
                    outp = plots_dir / corr_name
                    plt.savefig(str(outp), dpi=180, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    outs.append(str(outp))
            except Exception:
                pass
except Exception as e:
    import traceback as _tb
    print('ERROR: heatmap rendering failed:', e)
    _tb.print_exc()

# FeatureProfileSummary → bar charts
try:
    if __WANT_BARS__:
        summ = (payload.get('feature_profile') or {}).get('summary') or {}
        top_map = (summ.get('top') or {})
        if top_map:
            for ft, items in top_map.items():
                print(f"Bars[{ft}]: n={len(items)}")
                tag = _tag(f"bars_{ft}")
                fname = f"bars_top_{ft}_{tag}.png"
                use_names = (ft == 'pfam')
                bar = _save_bars(items, f"Top {ft.upper()} features (global totals)", fname, limit=10, use_names=use_names)
                if bar: outs.append(bar)
        else:
            # Backward-compat keys
            for ft, key in [('pfam','top_pfam'), ('ko','top_ko')]:
                items = summ.get(key) or []
                if items:
                    print(f"Bars[{ft}]: n={len(items)}")
                    tag = _tag(f"bars_{ft}")
                    fname = f"bars_top_{ft}_{tag}.png"
                    use_names = (ft == 'pfam')
                    bar = _save_bars(items, f"Top {ft.upper()} features (global totals)", fname, limit=10, use_names=use_names)
                    if bar: outs.append(bar)
        # Stacked proportions per genome (per feature type), only when composition cues appear
        m = (payload.get('feature_profile') or {}).get('per_genome_top_matrix') or {}
        order = (m.get('feature_order') or {})
        ql = (question_text or '').lower()
        want_stack = any(k in ql for k in ('proportion','composition','relative','share'))
        if want_stack:
            for ft in list((order or {}).keys()):
                rows = m.get(ft) or []
                cols = order.get(ft) or []
                if not rows or not cols:
                    continue
                import numpy as _np
                data = _np.array([[int(r.get(c,0)) for c in cols] for r in rows], dtype=float)
                if data.size == 0 or data.sum() <= 0:
                    continue
                totals = data.sum(axis=1, keepdims=True)
                totals[totals==0] = 1
                P = data / totals
                # pick top-K features by global weight, group the rest as 'other'
                k = min(8, P.shape[1])
                weights = P.sum(axis=0)
                idx = _np.argsort(-weights)[:k]
                keep_cols = [cols[i] for i in idx]
                other = P[:, [i for i in range(P.shape[1]) if i not in idx]].sum(axis=1)
                # plot
                plt.figure(figsize=(max(7, len(rows)*1.2), 5))
                bottom = _np.zeros(P.shape[0])
                x = _np.arange(P.shape[0])
                palette = ['#4c78a8','#f58518','#e45756','#72b7b2','#54a24b','#eeca3b','#b279a2','#ff9da6','#9d755d']
                for j, col_id in enumerate(keep_cols):
                    y = P[:, idx[j]]
                    plt.bar(x, y, bottom=bottom, color=palette[j % len(palette)], label=_abbrev_label(col_id, 12))
                    bottom += y
                if other.sum() > 1e-9:
                    plt.bar(x, 1-bottom, bottom=bottom, color='#bab0ac', label='other')
                plt.xticks(x, [r.get('genome_id') for r in rows], rotation=45, ha='right', fontsize=9)
                plt.ylabel('proportion', fontsize=10)
                plt.title(f"Per-genome composition (top {k} {ft.upper()} features)", fontsize=11)
                plt.legend(loc='upper left', bbox_to_anchor=(1.02,1), borderaxespad=0., fontsize=9)
                plt.tight_layout()
                sp_name = f"stacked_proportions_{ft}_{_tag('stack')}.png"
                outp = plots_dir / sp_name
                plt.savefig(str(outp), dpi=180, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                outs.append(str(outp))
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
