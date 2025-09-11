#!/usr/bin/env python3
"""
MinCED CRISPR parser and JSON synthesizer.

Converts MinCED outputs (GFF + spacers FASTA + optional .crisprs text)
into Stage 05 JSON artifacts expected by the Stage 07 KG builder:
  - <genome_id>_crispr_arrays.json
  - crispr_summary.json
  - processing_manifest.json

Usage (parse existing outputs):
  python -m src.ingest.minced_crispr parse \
      --genome-id SRR6231169 \
      --gff data/stage05_crispr/SRR6231169.gff \
      --spacers data/stage05_crispr/SRR6231169_spacers.fa \
      --input-fasta data/stage00_prepared/SRR6231169.fasta \
      --outdir data/stage05_crispr

Notes
- The GFF produced by MinCED contains one row per CRISPR array (type=repeat_region),
  with attributes including `ID=CRISPRn` and `rpt_unit_seq=...`.
- Column 6 (score) carries the number of repeats in the array.
- Spacers FASTA contains entries named like `<contig>_CRISPR_<n>_spacer_<i>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any


@dataclass
class CrisprArray:
    id: str
    genomeId: str
    contig: str
    startCoordinate: int
    endCoordinate: int
    repeatConsensus: Optional[str]
    repeatLength: Optional[int]
    repeatsCount: int
    spacerCount: int
    evidence: str = "minced"
    toolVersion: Optional[str] = None

    # Optional audit fields (not used by KG import, but helpful for debugging)
    spacerHeaders: Optional[List[str]] = None


def _stable_array_id(genome_id: str, contig: str, array_num: str) -> str:
    base = f"{genome_id}|{contig}|CRISPR{array_num}"
    # Avoid excessively long IDs in case contig names are huge
    if len(base) <= 240:
        return base
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{genome_id}|sha1:{h}|CRISPR{array_num}"


def parse_minced_gff(gff_path: Path, genome_id: str) -> Tuple[List[CrisprArray], Optional[str]]:
    """Parse MinCED GFF file.

    Returns a list of CrisprArray objects with repeatsCount and repeatConsensus;
    spacerCount remains to be filled (from spacers FASTA) and defaults to max(repeatsCount-1, 0).
    Also returns detected tool version string if present in the source column (e.g., 'minced:0.4.2').
    """
    arrays: List[CrisprArray] = []
    seen: set[tuple[str, str, int, int]] = set()
    version: Optional[str] = None
    # GFF columns: seqid, source, type, start, end, score, strand, phase, attributes
    with open(gff_path, "r") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype != "repeat_region":
                continue
            if source and source.startswith("minced:"):
                version = source.split(":", 1)[-1]
            # Attributes example: ID=CRISPR9;rpt_type=direct;rpt_family=CRISPR;rpt_unit_seq=...
            attr_map = {}
            for kv in attrs.split(";"):
                if not kv:
                    continue
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    attr_map[k] = v
            array_num = attr_map.get("ID", "CRISPR?").replace("CRISPR", "")
            rpt_seq = attr_map.get("rpt_unit_seq")
            repeats_count = 0
            try:
                repeats_count = int(score) if score and score != "." else 0
            except ValueError:
                repeats_count = 0
            start_i = int(start)
            end_i = int(end)
            arr_id = _stable_array_id(genome_id, seqid, array_num)
            key = (seqid, array_num, start_i, end_i)
            if key in seen:
                continue
            seen.add(key)
            arrays.append(
                CrisprArray(
                    id=arr_id,
                    genomeId=genome_id,
                    contig=seqid,
                    startCoordinate=start_i,
                    endCoordinate=end_i,
                    repeatConsensus=rpt_seq,
                    repeatLength=len(rpt_seq) if rpt_seq else None,
                    repeatsCount=repeats_count,
                    spacerCount=max(repeats_count - 1, 0),
                    toolVersion=version,
                )
            )
    return arrays, version


def parse_spacers_fasta(spacers_path: Path) -> Dict[Tuple[str, str], List[str]]:
    """Group spacer headers and sequences by (contig, array_num).

    Header format observed: >CONTIG_CRISPR_<n>_spacer_<i>
    Returns mapping to a list of spacer headers (sequence strings not required for KG).
    """
    mapping: Dict[Tuple[str, str], List[str]] = {}
    if not spacers_path.exists() or spacers_path.stat().st_size == 0:
        return mapping

    header_re = re.compile(r"^(?P<contig>.+)_CRISPR_(?P<num>\d+)_spacer_\d+$")
    current_header: Optional[str] = None
    with open(spacers_path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                m = header_re.match(header)
                if m:
                    key = (m.group("contig"), m.group("num"))
                    mapping.setdefault(key, []).append(header)
                current_header = header
            else:
                # sequence line; we don't need it for KG, skip
                pass
    return mapping


def synthesize_crispr_json(
    genome_id: str,
    input_fasta: Path,
    arrays: List[CrisprArray],
    spacers_by_array: Dict[Tuple[str, str], List[str]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Attach spacer counts/headers
    for ca in arrays:
        key = (ca.contig, ca.id.split("|")[-1].replace("CRISPR", ""))
        # safer key derivation by parsing the id tail; if mismatch, fallback to contig+num from id
        # Update spacer count when available
        # Attempt alternative parse if previous key fails
        if key not in spacers_by_array:
            # try to extract contig and num by splitting id: genome|contig|CRISPRn
            parts = ca.id.split("|")
            if len(parts) >= 3:
                contig = parts[-2]
                num = parts[-1].replace("CRISPR", "")
                key = (contig, num)
        if key in spacers_by_array:
            headers = spacers_by_array[key]
            ca.spacerCount = len(headers)
            ca.spacerHeaders = headers

    arrays_sorted = sorted(arrays, key=lambda a: (a.contig, a.startCoordinate))
    payload = {
        "version": "1.0.0",
        "stage": "stage05_crispr",
        "tool": "minced",
        "toolVersion": arrays[0].toolVersion if arrays else None,
        "genome_id": genome_id,
        "input_fasta": str(input_fasta),
        "arrays": [asdict(a) for a in arrays_sorted],
        "summary": {
            "total_arrays": len(arrays_sorted),
            "total_spacers": sum(a.spacerCount for a in arrays_sorted),
        },
    }

    with open(outdir / f"{genome_id}_crispr_arrays.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Summary across genomes (single genome here; leave for later aggregation)
    summary_file = outdir / "crispr_summary.json"
    summary = {
        "genomes": {
            genome_id: {
                "arrays": len(arrays_sorted),
                "spacers": sum(a.spacerCount for a in arrays_sorted),
            }
        }
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "stage": "stage05_crispr",
        "version": "1.0.0",
        "genomes": [genome_id],
        "inputs": [str(input_fasta)],
        "artifacts": [
            f"{genome_id}_crispr_arrays.json",
            "crispr_summary.json",
        ],
    }
    with open(outdir / "processing_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def cli_parse(args: argparse.Namespace) -> int:
    gff = Path(args.gff)
    spacers = Path(args.spacers) if args.spacers else Path("")
    outdir = Path(args.outdir)
    genome_id = args.genome_id
    input_fasta = Path(args.input_fasta) if args.input_fasta else gff.with_suffix("")

    arrays, version = parse_minced_gff(gff, genome_id)
    spacer_map = parse_spacers_fasta(spacers) if spacers and spacers.exists() else {}
    synthesize_crispr_json(genome_id, input_fasta, arrays, spacer_map, outdir)
    print(f"Parsed {len(arrays)} arrays; wrote JSON to {outdir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MinCED CRISPR parser → JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Parse existing GFF + spacers into JSON")
    p_parse.add_argument("--genome-id", required=True)
    p_parse.add_argument("--gff", required=True)
    p_parse.add_argument("--spacers", required=False, default=None)
    p_parse.add_argument("--input-fasta", required=False, default=None)
    p_parse.add_argument("--outdir", required=True)
    p_parse.set_defaults(func=cli_parse)

    # Batch runner: run MinCED and synthesize JSON for all FASTA files
    p_run = sub.add_parser("run", help="Run MinCED over input FASTA(s) and synthesize JSON")
    p_run.add_argument("--input-dir", required=True, help="Directory with *.fasta|*.fa|*.fna (stage00_prepared)")
    p_run.add_argument("--output-dir", required=True, help="Output directory (e.g., data/stage05_crispr)")
    p_run.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p_run.set_defaults(func=cli_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------- Pipeline-facing helpers ----------------

def run_minced_on_fasta(fasta: Path, outdir: Path, force: bool = False) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Run minced -gff -spacers on a single FASTA and return output paths.

    Returns (gff_path, spacers_path, txt_path); entries may be None on failure.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    genome_id = fasta.stem
    gff = outdir / f"{genome_id}.gff"
    spacers = outdir / f"{genome_id}_spacers.fa"
    txt = outdir / f"{genome_id}.crisprs"
    if force:
        # Ensure clean outputs on force
        for f in (gff, spacers, txt):
            try:
                if f and Path(f).exists():
                    Path(f).unlink()
            except Exception:
                pass
    if not force and gff.exists() and spacers.exists():
        return gff, spacers, (txt if txt.exists() else None)
    cmd = ["minced", "-gff", "-spacers", str(fasta), str(txt), str(gff)]
    try:
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if res.returncode != 0:
            print(f"[minced] Failed for {fasta.name}: {res.stderr}")
            return None, None, None
        else:
            print(f"[minced] {fasta.name} -> {gff.name} in {dt:.1f}s")
            return gff, spacers, txt
    except FileNotFoundError:
        print("[minced] 'minced' not found on PATH. Install via conda.")
        return None, None, None


def run_minced_batch(input_dir: Path, output_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Run minced for all FASTA files in input_dir and synthesize JSON artifacts."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_files = [p for p in input_dir.iterdir() if p.suffix.lower() in (".fasta", ".fa", ".fna")]
    results: Dict[str, int] = {}
    for fa in fasta_files:
        gid = fa.stem
        gff, spacers, _ = run_minced_on_fasta(fa, output_dir, force=force)
        if gff is None:
            continue
        arrays, _ver = parse_minced_gff(gff, gid)
        spacer_map = parse_spacers_fasta(spacers) if spacers and spacers.exists() else {}
        synthesize_crispr_json(gid, fa, arrays, spacer_map, output_dir)
        results[gid] = len(arrays)
    return {"genomes": len(results), "arrays_total": sum(results.values()), "details": results}


def cli_run(args: argparse.Namespace) -> int:
    stats = run_minced_batch(Path(args.input_dir), Path(args.output_dir), force=bool(args.force))
    print(f"MinCED batch complete: {stats}")
    return 0
