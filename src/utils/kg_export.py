from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ..build_kg.neo4j_bulk_loader import get_post_import_statements


def _sha256_file(p: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _line_count(p: Path) -> int:
    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _collect_csv_counts(csv_dir: Path) -> Tuple[int, int]:
    if not csv_dir.exists():
        return 0, 0
    node_files = [p for p in csv_dir.glob('*.csv') if 'relationships' not in p.name]
    rel_files = [p for p in csv_dir.glob('*.csv') if 'relationships' in p.name]
    def _rows(files):
        total = 0
        for f in files:
            n = _line_count(f)
            total += max(0, n - 1)  # minus header
        return total
    return _rows(node_files), _rows(rel_files)


def _write_post_import_cypher(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cy = out_dir / 'post_import.cypher'
    stmts = get_post_import_statements()
    cy.write_text('\n'.join(s + ';' for s in stmts) + '\n', encoding='utf-8')
    return cy


def _write_restore_scripts(bundle: Path, db_name: str = 'neo4j', image: str = None) -> None:
    scripts = bundle / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    if image is None:
        image = os.getenv('GENOME_KG_NEO4J_IMAGE', 'neo4j:5')
    restore_docker = scripts / 'restore_docker.sh'
    restore_docker.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$DIR/dumps/neo4j-5.x/neo4j.dump" ]; then
  docker run --rm -v "$DIR/dumps/neo4j-5.x":/import {image} neo4j-admin database load {db_name} --from-path=/import
fi
docker run -d --name kg-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=${{NEO4J_AUTH:-none}} -v "$DIR/data":/data {image}
echo "Bolt: bolt://localhost:7687"
""", encoding='utf-8')
    os.chmod(restore_docker, 0o755)

    restore_system = scripts / 'restore_system.sh'
    restore_system.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")/.." && pwd)
neo4j-admin database load {db_name} --from-path "$DIR/dumps/neo4j-5.x"
neo4j start
""", encoding='utf-8')
    os.chmod(restore_system, 0o755)

    # README
    readme = bundle / 'README.md'
    readme.write_text("""# KG Bundle

Artifacts:
- dumps/neo4j-5.x/neo4j.dump (primary)
- csv/ (optional) + scripts/post_import.cypher

Restore (Docker):
  scripts/restore_docker.sh

Restore (System):
  scripts/restore_system.sh
""", encoding='utf-8')

    # Smoke test script
    smoke = scripts / 'smoke_test_templates.py'
    smoke.write_text("""#!/usr/bin/env python3
from neo4j import GraphDatabase
import os

uri = os.getenv('NEO4J_URI','bolt://localhost:7687')
auth = None
u = os.getenv('NEO4J_USER')
p = os.getenv('NEO4J_PASSWORD')
if u and p:
    auth=(u,p)
drv = GraphDatabase.driver(uri, auth=auth)
with drv.session() as s:
    c = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print("nodes:", c)
    ca = s.run("MATCH (ca:CrisprArray) RETURN count(ca) AS c").single()["c"]
    print("crispr_arrays:", ca)
    genes = s.run("MATCH (g:Gene) RETURN count(g) AS c").single()["c"]
    print("genes:", genes)
drv.close()
print("OK")
""", encoding='utf-8')
    os.chmod(smoke, 0o755)


def export_bundle(out_dir: Path, fmt: str = 'dump', engine: str = 'docker') -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        'spec_version': 'kg-bundle/1',
        'created_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'database': {'name': 'neo4j'},
        'neo4j': {'image': os.getenv('GENOME_KG_NEO4J_IMAGE', 'neo4j:5'), 'major': 5},
        'artifacts': {},
    }

    # Optional provenance
    ttl = Path('data/stage07_kg/knowledge_graph.ttl')
    if ttl.exists():
        (out_dir / 'ttl').mkdir(exist_ok=True)
        dst = out_dir / 'ttl' / 'knowledge_graph.ttl'
        shutil.copy2(ttl, dst)
        manifest.setdefault('artifacts', {})['ttl'] = {
            'path': str(dst.relative_to(out_dir)),
            'sha256': _sha256_file(dst),
            'size_bytes': dst.stat().st_size,
        }

    if fmt in ('dump', 'both'):
        # dump via docker or system
        dump_dir = out_dir / 'dumps' / 'neo4j-5.x'
        dump_dir.mkdir(parents=True, exist_ok=True)
        if engine == 'docker':
            # Use offline store at data/neo4j
            store = Path('data/neo4j')
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{store.resolve()}:/data',
                '-v', f'{dump_dir.resolve()}:/out',
                os.getenv('GENOME_KG_NEO4J_IMAGE', 'neo4j:5'),
                'neo4j-admin', 'database', 'dump', 'neo4j', '--to-path=/out'
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f'docker dump failed: {res.stderr}')
        else:
            # system neo4j-admin
            cmd = ['neo4j-admin', 'database', 'dump', 'neo4j', f'--to-path={dump_dir}']
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f'system dump failed: {res.stderr}')
        dump_file = dump_dir / 'neo4j.dump'
        manifest['artifacts']['dump'] = {
            'path': str(dump_file.relative_to(out_dir)),
            'sha256': _sha256_file(dump_file),
            'size_bytes': dump_file.stat().st_size,
        }

    if fmt in ('csv', 'both'):
        csv_src = Path('data/stage07_kg/csv')
        csv_dst = out_dir / 'csv'
        if not csv_src.exists():
            raise FileNotFoundError('CSV source not found at data/stage07_kg/csv')
        if csv_dst.exists():
            shutil.rmtree(csv_dst)
        shutil.copytree(csv_src, csv_dst)
        # Write post_import.cypher
        cy = _write_post_import_cypher(out_dir / 'scripts')
        nodes, rels = _collect_csv_counts(csv_dst)
        manifest['artifacts']['csv'] = {
            'path': str(csv_dst.relative_to(out_dir)),
            'node_rows': nodes,
            'relationship_rows': rels,
        }
        manifest['counts'] = {'nodes': nodes, 'relationships': rels}
        manifest['artifacts']['post_import'] = {'path': str(cy.relative_to(out_dir))}

    # Write scripts
    _write_restore_scripts(out_dir)

    # Resolve dataset id heuristically (contig or genome id from CSV dir name)
    manifest['dataset_id'] = Path.cwd().name
    # Git info (best-effort)
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        if rev.returncode == 0:
            manifest['git'] = {'commit': rev.stdout.strip()}
    except Exception:
        pass

    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
