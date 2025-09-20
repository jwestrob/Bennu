#!/usr/bin/env python3
"""
Test runner: KEGG pathway completeness via Code Interpreter

This script:
  1) Serves the repo directory over HTTP so the interpreter can fetch ko_pathway.list
  2) Calls the interpreter's /execute endpoint with a self-contained script that:
       - Queries Neo4j (HTTP) for present KO IDs per genome
       - Fetches ko_pathway.list over HTTP
       - Computes per-pathway completeness and prints results

Prereqs:
  - Code interpreter container running (e.g., name 'code_interpreter')
  - Neo4j HTTP reachable from the container (macOS: host.docker.internal:7474)
  - Env NEO4J_PASSWORD set (or pass --neo4j-pass)

Usage:
  python scripts/pathway_completeness_via_interpreter.py \
    --interpreter http://localhost:8000 \
    --neo4j-url http://host.docker.internal:7474/db/neo4j/tx/commit \
    --neo4j-user neo4j \
    --neo4j-pass "$NEO4J_PASSWORD"
"""

from __future__ import annotations
import argparse
import contextlib
import json
import os
import socket
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

import requests


def _pick_port(preferred: int = 9000, fallback: int = 9001) -> int:
    for p in (preferred, fallback):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free HTTP port found for local test server (tried 9000, 9001)")


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # quiet
        pass


def _start_http_server(root: Path, port: int) -> ThreadingHTTPServer:
    # Serve the repo root so /data/reference/ko_pathway.list exists at the URL
    handler = lambda *args, **kwargs: _Handler(*args, directory=str(root), **kwargs)  # type: ignore
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KEGG pathway completeness via code interpreter")
    parser.add_argument("--interpreter", default="http://localhost:8000", help="Base URL for code interpreter service")
    parser.add_argument("--neo4j-url", default=os.getenv("NEO4J_HTTP_URL", "http://host.docker.internal:7474/db/neo4j/tx/commit"), help="Neo4j HTTP transactional endpoint")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j user")
    parser.add_argument("--neo4j-pass", default=os.getenv("NEO4J_PASSWORD", "password"), help="Neo4j password")
    parser.add_argument("--embed-ko", action="store_true", help="Embed ko_pathway.list inline (gzip+base64) instead of serving over HTTP")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ko_file = repo_root / "data" / "reference" / "ko_pathway.list"
    if not ko_file.exists():
        print(f"ERROR: Missing file: {ko_file}")
        return 2

    httpd = None
    if args.embed_ko:
        # Read and embed ko_pathway.list (gzip+base64)
        import gzip, base64
        ko_raw = ko_file.read_bytes()
        ko_b64 = base64.b64encode(gzip.compress(ko_raw)).decode('ascii')
        ko_decl = f"KO_LIST_B64 = '{ko_b64}'\nEMBEDDED_KO = True\n"
        ko_fetch = """
import base64, gzip
ko_text = gzip.decompress(base64.b64decode(KO_LIST_B64)).decode('utf-8')
"""
    else:
        port = _pick_port(9000, 9001)
        httpd = _start_http_server(repo_root, port)
        time.sleep(0.3)
        ko_url = f"http://host.docker.internal:{port}/data/reference/ko_pathway.list"
        ko_decl = f"KO_LIST_URL = {json.dumps(ko_url)}\nEMBEDDED_KO = False\n"
        ko_fetch = """
import requests
r2 = requests.get(KO_LIST_URL, timeout=30)
r2.raise_for_status()
ko_text = r2.text
"""

    code = f"""
import json, requests
from collections import defaultdict

NEO4J_HTTP_URL = {json.dumps(args.neo4j_url)}
NEO4J_USER = {json.dumps(args.neo4j_user)}
NEO4J_PASSWORD = {json.dumps(args.neo4j_pass)}
{ko_decl}

payload = {{"statements":[{{"statement": '''
MATCH (g:Genome)<-[:BELONGSTOGENOME]-(gene:Gene)-[:ENCODEDBY]->(:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
RETURN g.id AS genome_id, collect(DISTINCT ko.id) AS present_ko_ids
''', "parameters": {{}}}}]}}

r = requests.post(NEO4J_HTTP_URL, json=payload, auth=(NEO4J_USER, NEO4J_PASSWORD), timeout=30)
r.raise_for_status()
j = r.json()
if j.get('errors'):
    raise RuntimeError(j['errors'])
rows = j['results'][0]['data']

present = {{}}
for d in rows:
    row = d.get('row', [])
    if len(row) >= 2:
        gid = str(row[0])
        kos = [str(k) for k in (row[1] or [])]
        present[gid] = kos

{ko_fetch}
pathway_kos = defaultdict(set)
for line in ko_text.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    ko, path = parts
    ko = ko.replace('ko:', '')
    path = path.replace('path:', '')
    if not path.startswith('map'):
        continue
    pathway_kos[path].add(ko)

def completeness_rows(genome_kos):
    s = set(genome_kos)
    rows = []
    for pw, all_kos in pathway_kos.items():
        total = len(all_kos)
        if total == 0:
            continue
        pc = len(all_kos & s)
        comp = pc / total
        rows.append((pw, pc, total, comp))
    rows.sort(key=lambda x: (-x[3], -x[1], x[0]))
    return rows

for gid, kos in present.items():
    print(f"Genome: {{gid}}")
    r = completeness_rows(kos)
    if not r:
        print("  No pathways evaluated.")
        continue
    complete = [t for t in r if abs(t[3] - 1.0) < 1e-9]
    show = complete if complete else r[:25]
    for pw, pc, tot, comp in show:
        print(f"  {{pw}}: {{pc}}/{{tot}} ({{comp:.2f}})")
    print()
"""

    payload = {"session_id": "pathway-completeness", "timeout": 60, "code": code}
    url = args.interpreter.rstrip("/") + "/execute"
    try:
        resp = requests.post(url, json=payload, timeout=90)
        print(resp.status_code)
        data = resp.json()
        out = data.get("stdout", "")
        err = data.get("stderr", "")
        if out:
            print(out)
        if err:
            print("STDERR:\n" + err)
    finally:
        with contextlib.suppress(Exception):
            httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
