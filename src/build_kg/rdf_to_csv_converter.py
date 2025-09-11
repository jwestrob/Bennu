#!/usr/bin/env python3
"""
Convert RDF knowledge graph to Neo4j-compatible CSV files for bulk import.
Designed for 100x faster loading than current Python-based approach.
"""

import csv
import json
import bisect
import rdflib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple
import logging
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)


class RDFToCSVConverter:
    """Convert RDF triples to Neo4j bulk import CSV format."""
    
    def __init__(self, rdf_file: Path, output_dir: Path):
        self.rdf_file = rdf_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define namespaces
        self.namespaces = {
            "genome": "http://genome-kg.org/genomes/",
            "protein": "http://genome-kg.org/proteins/", 
            "gene": "http://genome-kg.org/genes/",
            "pfam": "http://pfam.xfam.org/family/",
            "ko": "http://www.genome.jp/kegg/ko/",
            "kg": "http://genome-kg.org/ontology/",
            "cazyme": "http://genome-kg.org/cazyme/",
            "pathway": "http://genomics.ai/kg/pathway/"
        }
        
        # Track all nodes and their properties
        self.nodes = defaultdict(dict)  # {node_id: {property: value}}
        self.node_types = {}  # {node_id: type}
        self.relationships = []  # [(from_id, rel_type, to_id)]
        
        # Known type → filename mapping to preserve legacy names
        self.type_filename = {
            'Genome': 'genomes.csv',
            'Gene': 'genes.csv',
            'Protein': 'proteins.csv',
            'FunctionalAnnotation': 'functionalannotations.csv',
            'DomainAnnotation': 'domainannotations.csv',
            'KEGGOrtholog': 'keggorthologs.csv',
            'Domain': 'domains.csv',
            'Pathway': 'pathways.csv',
            'Bgc': 'bgcs.csv',
            'QualityMetrics': 'qualitymetrics.csv',
            'Contig': 'contigs.csv',
            'Dataset': 'datasets.csv',
            # Do not emit a separate Entity file; it's an abstract base
        }
        
    def convert(self) -> Dict[str, Any]:
        """Convert RDF to CSV files and return statistics."""
        console.print(f"[bold blue]Converting RDF to CSV for bulk import[/bold blue]")
        
        # Load RDF
        console.print("Loading RDF graph...")
        g = rdflib.Graph()
        loaded = False
        for fmt in ("nt", "turtle", "xml"):
            try:
                g.parse(self.rdf_file, format=fmt)
                console.print(f"Loaded {len(g):,} triples (format={fmt})")
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            raise RuntimeError(f"Failed to parse RDF file {self.rdf_file} in known formats (nt, turtle, xml)")
        
        # Parse triples
        self._parse_triples(g)
        
        # Clean stale CSVs from previous runs in this directory
        self._clean_output_dir()

        # Integrate CRISPR arrays (if present) and precompute enhanced NEXT edges
        crispr_stats = self._integrate_crispr_and_neighbors()

        # Write CSV files
        stats = self._write_csv_files(crispr_stats)
        
        console.print(f"[green]✓ CSV conversion complete![/green]")
        return stats
    
    def _parse_triples(self, g: rdflib.Graph):
        """Parse RDF triples into nodes and relationships."""
        console.print("Parsing triples...")
        
        for subj, pred, obj in track(g, description="Processing triples"):
            subj_id = self._uri_to_id(str(subj))
            pred_name = self._uri_to_property(str(pred))
            
            # Handle rdf:type declarations
            if str(pred) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                node_type = self._get_node_type(str(obj))
                # Skip ontology + abstract base 'Entity' to preserve legacy CSV set
                if node_type not in ["Property", "Class", "Entity"]:
                    self.node_types[subj_id] = node_type
                continue
            
            # Handle relationships vs properties
            if isinstance(obj, rdflib.URIRef):
                # This is a relationship
                obj_id = self._uri_to_id(str(obj))
                self.relationships.append((subj_id, pred_name, obj_id))
            else:
                # This is a node property
                obj_value = self._convert_literal(obj)
                self.nodes[subj_id][pred_name] = obj_value
    
    def _write_csv_files(self, crispr_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Write separate CSV files for each node type and relationships."""
        stats = {"nodes": {}, "relationships": {}}
        
        # Group nodes by type
        nodes_by_type = defaultdict(list)
        for node_id, node_type in self.node_types.items():
            nodes_by_type[node_type].append(node_id)
        
        # Write node CSV files
        console.print("Writing node CSV files...")
        for node_type, node_ids in nodes_by_type.items():
            if not node_ids:
                continue
            # Preserve legacy file names when known
            filename = self.type_filename.get(node_type)
            if not filename:
                # Fallback: simple pluralization with -y -> -ies
                base_name = node_type.lower()
                if base_name.endswith('y'):
                    filename = f"{base_name[:-1]}ies.csv"
                elif base_name.endswith('s'):
                    filename = f"{base_name}.csv"
                else:
                    filename = f"{base_name}s.csv"
            
            filepath = self.output_dir / filename
            
            # Collect all possible properties for this node type
            all_properties = set()
            for node_id in node_ids:
                all_properties.update(self.nodes[node_id].keys())
            
            # Keep all properties - we don't want to remove legitimate properties
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                # Neo4j bulk import expects ID column to be labeled with :ID
                # Use a unique name that won't conflict with properties
                id_column = 'id:ID'
                fieldnames = [id_column] + sorted(all_properties)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for node_id in node_ids:
                    row = {id_column: node_id}
                    row.update(self.nodes[node_id])
                    writer.writerow(row)
            
            stats["nodes"][node_type] = len(node_ids)
            console.print(f"  ✓ {filename}: {len(node_ids):,} nodes")
        
        # Write CRISPR array nodes if present
        crispr_nodes: List[Dict[str, Any]] = crispr_stats.get("crispr_nodes") or []
        if crispr_nodes:
            filepath = self.output_dir / "crispr_arrays.csv"
            fieldnames = ["id:ID", "genomeId", "contig", "startCoordinate:int", "endCoordinate:int", "repeatConsensus", "repeatLength:int", "repeatsCount:int", "spacerCount:int", "evidence", "toolVersion"]
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in crispr_nodes:
                    out = {
                        "id:ID": row.get("id"),
                        "genomeId": row.get("genomeId"),
                        "contig": row.get("contig"),
                        "startCoordinate:int": int(row.get("startCoordinate") or 0),
                        "endCoordinate:int": int(row.get("endCoordinate") or 0),
                        "repeatConsensus": row.get("repeatConsensus"),
                        "repeatLength:int": int(row.get("repeatLength") or 0) if row.get("repeatLength") is not None else 0,
                        "repeatsCount:int": int(row.get("repeatsCount") or 0),
                        "spacerCount:int": int(row.get("spacerCount") or 0),
                        "evidence": row.get("evidence"),
                        "toolVersion": row.get("toolVersion"),
                    }
                    writer.writerow(out)
            stats["nodes"]["CrisprArray"] = len(crispr_nodes)
            console.print(f"  ✓ crispr_arrays.csv: {len(crispr_nodes):,} nodes")
        
        # Adjacency/degree now emitted in RDF (Option A); do not derive here

        # Write relationship CSV files from RDF triples
        console.print("Writing relationship CSV files...")
        rels_by_type = defaultdict(list)
        for from_id, rel_type, to_id in self.relationships:
            rels_by_type[rel_type].append((from_id, to_id))
        
        for rel_type, rels in rels_by_type.items():
            # Skip NEXT from RDF; we will write enhanced next_relationships.csv below
            if rel_type.upper() == "NEXT":
                continue
            filename = f"{rel_type.lower()}_relationships.csv"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([':START_ID', ':END_ID'])  # Neo4j bulk import format
                writer.writerows(rels)
            
            stats["relationships"][rel_type] = len(rels)
            console.print(f"  ✓ {filename}: {len(rels):,} relationships")
        
        # Write CRISPR relationships (BELONGSTOGENOME, FLANKS_CRISPR)
        crispr_belong_rels: List[Tuple[str, str]] = crispr_stats.get("crispr_belong_rels") or []
        if crispr_belong_rels:
            # Append CRISPR BELONGSTOGENOME rows to the existing file emitted from RDF.
            filepath = self.output_dir / "belongstogenome_relationships.csv"
            header_needed = not filepath.exists() or filepath.stat().st_size == 0
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow([':START_ID', ':END_ID'])
                writer.writerows(crispr_belong_rels)
            stats["relationships"]["BELONGSTOGENOME"] = stats["relationships"].get("BELONGSTOGENOME", 0) + len(crispr_belong_rels)
            console.print(f"  ✓ belongstogenome_relationships.csv (+{len(crispr_belong_rels):,} CRISPR rows)")
        crispr_flank_rels: List[Dict[str, Any]] = crispr_stats.get("crispr_flank_rels") or []
        if crispr_flank_rels:
            filepath = self.output_dir / "flanks_crispr_relationships.csv"
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([':START_ID', ':END_ID', 'side:string', 'distanceBp:int'])
                for r in crispr_flank_rels:
                    writer.writerow([r['start_id'], r['end_id'], r.get('side') or '', int(r.get('distanceBp') or 0)])
            stats["relationships"]["FLANKS_CRISPR"] = len(crispr_flank_rels)
            console.print(f"  ✓ flanks_crispr_relationships.csv: {len(crispr_flank_rels):,} relationships")
        
        # Write enhanced NEXT relationships
        next_rels: List[Dict[str, Any]] = crispr_stats.get("next_rels") or []
        if next_rels:
            filepath = self.output_dir / "next_relationships.csv"
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([':START_ID', ':END_ID', 'contig', 'delta:int', 'same_strand:boolean', 'crisprBetween:boolean', 'crisprCountBetween:int'])
                for r in next_rels:
                    def _b(v):
                        return 'true' if bool(v) else 'false'
                    writer.writerow([
                        r['start_id'], r['end_id'], r.get('contig') or '',
                        int(r.get('delta') or 0), _b(r.get('same_strand')),
                        _b(r.get('crisprBetween')), int(r.get('crisprCountBetween') or 0)
                    ])
            stats["relationships"]["NEXT"] = len(next_rels)
            console.print(f"  ✓ next_relationships.csv (enhanced): {len(next_rels):,} relationships")
        
        return stats
    
    def _uri_to_id(self, uri: str) -> str:
        """Convert URI to readable ID, preserving namespace for nodes to avoid conflicts."""
        for prefix, namespace in self.namespaces.items():
            if uri.startswith(namespace):
                local_id = uri.replace(namespace, "")
                # Preserve namespace prefix for nodes to distinguish protein:X from gene:X from cazyme:X
                if prefix in ['protein', 'gene', 'cazyme']:
                    return f"{prefix}:{local_id}"
                return local_id
        return uri.split("/")[-1]
    
    def _uri_to_property(self, uri: str) -> str:
        """Convert property URI to readable name."""
        if uri.startswith(self.namespaces["kg"]):
            return uri.replace(self.namespaces["kg"], "")
        return uri.split("/")[-1].split("#")[-1]
    
    def _get_node_type(self, type_uri: str) -> str:
        """Determine node type from RDF type URI."""
        if type_uri.startswith(self.namespaces["kg"]):
            return type_uri.replace(self.namespaces["kg"], "")
        return type_uri.split("/")[-1].split("#")[-1]
    
    def _convert_literal(self, literal: rdflib.Literal) -> Any:
        """Convert RDF literal to appropriate Python type."""
        if literal.datatype == rdflib.XSD.integer:
            return int(literal)
        elif literal.datatype in [rdflib.XSD.decimal, rdflib.XSD.double, rdflib.XSD.float]:
            return float(literal)
        elif literal.datatype == rdflib.XSD.boolean:
            return str(literal).lower() == 'true'
        else:
            return str(literal)

    def _clean_output_dir(self) -> None:
        """Remove stale CSVs to prevent mixing schemas across runs."""
        try:
            for f in self.output_dir.glob('*.csv'):
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    # ---------- CRISPR integration and NEXT computation ----------
    def _integrate_crispr_and_neighbors(self) -> Dict[str, Any]:
        """Load CRISPR JSON artifacts and compute enhanced NEXT edges and gene props.

        Returns dict with:
          - crispr_nodes: list of dicts for crispr_arrays.csv
          - crispr_belong_rels: list of (array_id, genome_id)
          - crispr_flank_rels: list of dicts with start_id, end_id, side, distanceBp
          - next_rels: list with enhanced NEXT properties including CRISPR crossings
        Also updates self.nodes for Gene nodes with nextDegree, genesOnContig, flank flags, nearestCrisprDistanceGenes.
        """
        # Build gene→genome map from RDF rels
        gene_to_genome: Dict[str, str] = {}
        for a, r, b in self.relationships:
            if r.upper() == 'BELONGSTOGENOME' and a.startswith('gene:'):
                gene_to_genome[a] = b

        # Collect genes grouped by (genome, contig) for adjacency
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for node_id, t in self.node_types.items():
            if t != 'Gene':
                continue
            props = self.nodes.get(node_id, {})
            contig = props.get('contig')
            start = props.get('startCoordinate')
            end = props.get('endCoordinate')
            strand = props.get('strand')
            if contig is None or start is None or end is None:
                continue
            genome = gene_to_genome.get(node_id, '')
            key = (genome, str(contig))
            groups[key].append({
                'id': node_id, 'start': int(start), 'end': int(end), 'strand': str(strand) if strand is not None else ''
            })

        # Load CRISPR arrays JSON files
        crispr_dir = Path('data/stage05_crispr')
        crispr_nodes_map: Dict[str, Dict[str, Any]] = {}
        crispr_belong_rels: List[Tuple[str, str]] = []
        crispr_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        if crispr_dir.exists():
            for json_file in crispr_dir.glob('*_crispr_arrays.json'):
                try:
                    data = json.loads(json_file.read_text())
                except Exception:
                    continue
                genome_id = str(data.get('genome_id') or data.get('genomeId') or '')
                arrays = data.get('arrays') or []
                for arr in arrays:
                    arr_id = arr.get('id')
                    contig = str(arr.get('contig') or '')
                    start = int(arr.get('startCoordinate') or 0)
                    end = int(arr.get('endCoordinate') or 0)
                    repeats = int(arr.get('repeatsCount') or 0)
                    spacers = int(arr.get('spacerCount') or 0)
                    rpt = arr.get('repeatConsensus')
                    rpt_len = int(arr.get('repeatLength') or (len(rpt) if rpt else 0))
                    tool_ver = arr.get('toolVersion')
                    node = {
                        'id': arr_id, 'genomeId': genome_id, 'contig': contig,
                        'startCoordinate': start, 'endCoordinate': end,
                        'repeatConsensus': rpt, 'repeatLength': rpt_len,
                        'repeatsCount': repeats, 'spacerCount': spacers,
                        'evidence': 'minced', 'toolVersion': tool_ver
                    }
                    # Deduplicate by id; if duplicate appears, keep the first
                    crispr_nodes_map.setdefault(arr_id, node)
                    crispr_belong_rels.append((arr_id, genome_id))
                    key = (genome_id, contig)
                    crispr_by_group[key].append({'start': start, 'end': end, 'id': arr_id})

        # Sort groups
        for key in groups:
            groups[key].sort(key=lambda x: (x['start'], x['end']))
        for key in crispr_by_group:
            arrs = sorted(crispr_by_group[key], key=lambda x: (x['start'], x['end']))
            seen_ids = set()
            dedup = []
            for a in arrs:
                if a['id'] in seen_ids:
                    continue
                seen_ids.add(a['id'])
                dedup.append(a)
            crispr_by_group[key] = dedup
        # Materialize unique CRISPR nodes list
        crispr_nodes: List[Dict[str, Any]] = list(crispr_nodes_map.values())

        # Compute NEXT pairs with CRISPR decorations + gene props
        next_rels: List[Dict[str, Any]] = []
        crispr_flank_rels: List[Dict[str, Any]] = []

        for (genome, contig), genes in groups.items():
            n = len(genes)
            # Update genesOnContig and nextDegree
            for i, g in enumerate(genes):
                node_id = g['id']
                self.nodes[node_id]['genesOnContig'] = int(n)
                deg = 0 if n == 1 else (1 if i in (0, n - 1) else 2)
                self.nodes[node_id]['nextDegree'] = int(deg)

            arrays = crispr_by_group.get((genome, contig), [])
            arr_starts = [a['start'] for a in arrays]
            # Compute NEXT
            for i in range(n - 1):
                a = genes[i]
                b = genes[i + 1]
                a_end = int(a['end'])
                b_start = int(b['start'])
                # determine gap interval [min_end, max_start]
                gap_start = a_end
                gap_end = b_start
                if gap_end < gap_start:
                    gap_start, gap_end = gap_end, gap_start
                # Count arrays fully within the gap
                count_between = 0
                if arrays:
                    # Narrow search by start coordinate using bisect
                    lo = bisect.bisect_left(arr_starts, gap_start)
                    hi = bisect.bisect_right(arr_starts, gap_end)
                    for j in range(max(lo - 2, 0), min(hi + 2, len(arrays))):
                        arr = arrays[j]
                        if arr['start'] >= gap_start and arr['end'] <= gap_end:
                            count_between += 1
                next_rels.append({
                    'start_id': a['id'], 'end_id': b['id'], 'contig': contig,
                    'delta': int(b_start) - int(a_end),
                    'same_strand': (a.get('strand') == b.get('strand') and a.get('strand') not in (None, '')),
                    'crisprBetween': count_between > 0,
                    'crisprCountBetween': int(count_between),
                })

            # Compute flank edges for arrays
            if arrays and genes:
                # List of gene starts and ends for quick searches
                gene_ends = [int(g['end']) for g in genes]
                gene_starts = [int(g['start']) for g in genes]
                for arr in arrays:
                    s, e = int(arr['start']), int(arr['end'])
                    # Left flank: rightmost gene with end <= start
                    li = bisect.bisect_right(gene_ends, s) - 1
                    if 0 <= li < n:
                        left_gene = genes[li]
                        crispr_flank_rels.append({
                            'start_id': left_gene['id'], 'end_id': arr['id'], 'side': 'left', 'distanceBp': max(0, s - int(left_gene['end']))
                        })
                        # mark flag
                        self.nodes[left_gene['id']]['isCrisprFlankLeft'] = True
                    # Right flank: leftmost gene with start >= end
                    ri = bisect.bisect_left(gene_starts, e)
                    if 0 <= ri < n:
                        right_gene = genes[ri]
                        crispr_flank_rels.append({
                            'start_id': right_gene['id'], 'end_id': arr['id'], 'side': 'right', 'distanceBp': max(0, int(right_gene['start']) - e)
                        })
                        self.nodes[right_gene['id']]['isCrisprFlankRight'] = True

                # nearestCrisprDistanceGenes
                arr_positions = [int((a['start'] + a['end']) // 2) for a in arrays]
                for i, g in enumerate(genes):
                    pos = (int(g['start']) + int(g['end'])) // 2
                    # Approximate distance in genes via nearest flank index
                    # Find closest array center by coordinate
                    if arr_positions:
                        ai = bisect.bisect_left(arr_positions, pos)
                        # nearest is min distance to either neighbor array boundary measured in genes via index distance
                        # As an approximation, use min distance to nearest flank gene index
                        # Compute distance to nearest array left or right flank gene index
                        nearest = n  # large
                        # left flank index (li) for array at ai-1
                        candidates = []
                        if ai - 1 >= 0:
                            arr_left = arrays[ai - 1]
                            li = bisect.bisect_right(gene_ends, int(arr_left['start'])) - 1
                            if 0 <= li < n:
                                candidates.append(abs(i - li))
                        if ai < len(arrays):
                            arr_right = arrays[ai]
                            ri = bisect.bisect_left(gene_starts, int(arr_right['end']))
                            if 0 <= ri < n:
                                candidates.append(abs(ri - i))
                        if candidates:
                            nearest = min(candidates)
                        else:
                            nearest = n
                        self.nodes[g['id']]['nearestCrisprDistanceGenes'] = int(nearest if nearest != n else -1)
                    else:
                        self.nodes[g['id']]['nearestCrisprDistanceGenes'] = -1

        # Dedup relationships to avoid duplicate lines
        crispr_belong_unique = list({(a,b) for (a,b) in crispr_belong_rels})
        flank_set = {(r['start_id'], r['end_id'], r.get('side') or '', int(r.get('distanceBp') or 0)) for r in crispr_flank_rels}
        crispr_flank_rels = [
            {'start_id': s, 'end_id': e, 'side': side, 'distanceBp': dist}
            for (s,e,side,dist) in flank_set
        ]

        return {
            'crispr_nodes': crispr_nodes,
            'crispr_belong_rels': crispr_belong_unique,
            'crispr_flank_rels': crispr_flank_rels,
            'next_rels': next_rels,
        }


def main():
    """Convert RDF to CSV for bulk import."""
    rdf_file = Path("data/stage07_kg/knowledge_graph.ttl")
    csv_dir = Path("data/stage07_kg/csv")
    
    if not rdf_file.exists():
        console.print(f"[red]RDF file not found: {rdf_file}[/red]")
        return 1
    
    converter = RDFToCSVConverter(rdf_file, csv_dir)
    stats = converter.convert()
    
    console.print(f"\n[bold]Conversion Summary:[/bold]")
    console.print(f"Output directory: {csv_dir}")
    
    total_nodes = sum(stats["nodes"].values())
    total_rels = sum(stats["relationships"].values())
    
    console.print(f"Total nodes: {total_nodes:,}")
    console.print(f"Total relationships: {total_rels:,}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
