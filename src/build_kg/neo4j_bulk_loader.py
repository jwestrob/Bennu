#!/usr/bin/env python3
"""
Bulk load Neo4j database using CSV files and neo4j-admin import.
Designed for 100x faster loading than Python-based MERGE operations.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
from rich.console import Console
from rich.progress import Progress
import sys
import json
import os

console = Console()


class Neo4jBulkLoader:
    """Bulk load Neo4j database using neo4j-admin import.

    Supports two engines:
    - system: use locally installed neo4j-admin (Homebrew/apt)
    - docker: run neo4j:5 container with NEO4J_AUTH=none; import via docker run and then start container
    """
    
    def __init__(self, csv_dir: Path, neo4j_home: Path = None, database_name: str = "neo4j", engine: str = "docker"):
        self.csv_dir = csv_dir
        self.database_name = database_name
        self.engine = engine.lower()
        self.data_dir = Path("data/neo4j").resolve()
        self.docker_image = os.getenv("GENOME_KG_NEO4J_IMAGE", "neo4j:5")
        self.container_name = os.getenv("GENOME_KG_NEO4J_CONTAINER", "genome-kg-neo4j")

        console.print(f"CSV directory: {self.csv_dir}")
        console.print(f"Engine: {self.engine}")

        if self.engine == "system":
            # Auto-detect Neo4j home from homebrew installation
            if neo4j_home is None:
                neo4j_home = self._detect_neo4j_home()
            self.neo4j_home = neo4j_home
            self.neo4j_admin = neo4j_home / "bin" / "neo4j-admin"
            self.neo4j_ctl = neo4j_home / "bin" / "neo4j"
            console.print(f"Neo4j home: {self.neo4j_home}")
    
    def _detect_neo4j_home(self) -> Path:
        """Auto-detect Neo4j installation directory."""
        # Try common homebrew locations
        candidates = [
            Path("/opt/homebrew/var/homebrew/linked/neo4j"),
            Path("/usr/local/var/homebrew/linked/neo4j"), 
            Path("/opt/homebrew/Cellar/neo4j").glob("*/libexec"),
            Path("/usr/local/Cellar/neo4j").glob("*/libexec")
        ]
        
        for candidate in candidates:
            if isinstance(candidate, Path):
                if (candidate / "bin" / "neo4j-admin").exists():
                    return candidate
            else:
                # Handle glob results
                for path in candidate:
                    if (path / "bin" / "neo4j-admin").exists():
                        return path
        
        # Try neo4j command in PATH
        try:
            result = subprocess.run(["which", "neo4j"], capture_output=True, text=True)
            if result.returncode == 0:
                neo4j_path = Path(result.stdout.strip())
                return neo4j_path.parent.parent  # Remove bin/neo4j to get home
        except:
            pass
        
        raise FileNotFoundError("Could not auto-detect Neo4j installation. Please specify neo4j_home parameter.")
    
    def bulk_import(self) -> Dict[str, Any]:
        """Perform complete bulk import process."""
        start_time = time.time()
        
        with Progress(console=console) as progress:
            task = progress.add_task("Bulk importing to Neo4j...", total=7)
            
            # Step 1: Validate CSV files
            progress.update(task, description="Validating CSV files...")
            csv_files = self._validate_csv_files()
            has_next_rel_csv = any('next_relationships.csv' in f.name.lower() for f in csv_files)
            progress.advance(task)
            
            if self.engine == "docker":
                # Step 2: Stop and remove any existing container
                progress.update(task, description="Stopping docker container...")
                self._docker_stop_container()
                progress.advance(task)

                # Step 3: Prepare data directory
                progress.update(task, description="Preparing data volume...")
                self.data_dir.mkdir(parents=True, exist_ok=True)
                progress.advance(task)

                # Step 4: Run import in docker
                progress.update(task, description="Running docker import...")
                import_stats = self._docker_run_import(csv_files)
                progress.advance(task)

                # Step 5: Start container
                progress.update(task, description="Starting docker Neo4j...")
                self._docker_start_container()
                progress.advance(task)

                # Step 6: Create constraints and indexes (skipped by default; can be triggered separately)
                progress.update(task, description="Creating constraints/indexes...")
                self._create_constraints_and_indexes()
                progress.advance(task)
            else:
                # system engine (existing behavior)
                # Step 2: Stop Neo4j
                progress.update(task, description="Stopping Neo4j...")
                self._stop_neo4j()
                progress.advance(task)
                
                # Step 3: Backup/clear existing database
                progress.update(task, description="Preparing database...")
                self._prepare_database()
                progress.advance(task)
                
                # Step 4: Run bulk import
                progress.update(task, description="Running bulk import...")
                import_stats = self._run_bulk_import(csv_files)
                progress.advance(task)
                
                # Step 5: Start Neo4j
                progress.update(task, description="Starting Neo4j...")
                self._start_neo4j()
                progress.advance(task)
                
                # Step 6: Create constraints and indexes
                progress.update(task, description="Creating constraints/indexes...")
                self._create_constraints_and_indexes()
                progress.advance(task)

            # Step 7: Precompute genomic neighbor edges
            if has_next_rel_csv:
                progress.update(task, description="Skipping neighbor precompute (NEXT CSV present)...")
                progress.advance(task)
            else:
                progress.update(task, description="Precomputing genomic neighbor edges...")
                self._precompute_neighbor_edges()
                progress.advance(task)
        
        total_time = time.time() - start_time
        
        console.print(f"[green]✓ Bulk import completed in {total_time:.2f} seconds![/green]")
        # Write connection.json for zero-config clients
        try:
            conn = {"uri": "bolt://localhost:7687", "auth": None}
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / "connection.json").write_text(json.dumps(conn))
        except Exception:
            pass
        
        return {
            "import_time_seconds": total_time,
            "csv_files_processed": len(csv_files),
            **import_stats
        }

    def _docker_available(self) -> bool:
        try:
            result = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _docker_stop_container(self):
        if not self._docker_available():
            raise RuntimeError("Docker is not available. Install Docker or use engine='system'.")
        # Stop and remove existing container if present
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, text=True)
    
    def _docker_run_import(self, csv_files: List[Path]) -> Dict[str, Any]:
        if not self._docker_available():
            raise RuntimeError("Docker is not available. Install Docker or use engine='system'.")
        # Ensure a clean destination store after previous failed imports
        try:
            store = self.data_dir / 'databases' / self.database_name
            if store.exists():
                shutil.rmtree(store)
        except Exception as e:
            console.print(f"[yellow]Warning: failed to clean existing store: {e}[/yellow]")
        # Build import command using /import mount
        node_files = []
        rel_files = []
        for csv_file in csv_files:
            if "relationships" in csv_file.name:
                rel_files.append(csv_file.name)
            else:
                node_files.append(csv_file.name)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.csv_dir.resolve()}:/import",
            "-v", f"{self.data_dir}:/data",
            self.docker_image,
            "neo4j-admin", "database", "import", "full",
            "--overwrite-destination=true",
            "--verbose",
        ]
        # Map filenames to container paths
        def label_for(stem_lower: str, filename: str) -> str:
            # Mimic system importer label overrides
            overrides = {
                "domainannotations": "DomainAnnotation",
                "functionalannotations": "FunctionalAnnotation",
                "keggorthologs": "KEGGOrtholog",
                "qualitymetrics": "QualityMetrics",
                "bgcs": "Bgc",
                "bgc_clusters": "Bgc",
                # CAZy
                "cazymeannotations": "Cazymeannotation",
                "cazymefamilies": "Cazymefamily",
                # CRISPR
                "crispr_arrays": "CrisprArray",
            }
            return overrides.get(stem_lower, filename.split(".")[0].rstrip('s').title())
        for node in node_files:
            stem_lower = Path(node).stem.lower()
            label = label_for(stem_lower, node)
            cmd.extend(["--nodes", f"{label}=/import/{node}"])
        for rel in rel_files:
            rel_type = Path(rel).stem.replace("_relationships", "").upper()
            cmd.extend(["--relationships", f"{rel_type}=/import/{rel}"])

        console.print("Running docker import...")
        console.print("Command: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            console.print(f"[red]Import failed![/red]")
            console.print(f"STDOUT: {result.stdout}")
            console.print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"neo4j-admin import (docker) failed: {result.stderr}")
        console.print("✓ Import successful!")
        console.print(f"Import output: {result.stdout}")
        return {"docker_import_output": result.stdout}

    def _docker_start_container(self):
        if not self._docker_available():
            raise RuntimeError("Docker is not available. Install Docker or use engine='system'.")
        # Start container with auth=none
        run_cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", "7474:7474", "-p", "7687:7687",
            "-e", "NEO4J_AUTH=none",
            "-v", f"{self.data_dir}:/data",
            self.docker_image
        ]
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start docker Neo4j: {result.stderr}")
        # Wait a bit for readiness
        console.print("Waiting for Neo4j (docker) to be ready...")
        time.sleep(5)
        # Poll bolt port indirectly by trying a few times
        for _ in range(30):
            # Try logs check (simple heuristic)
            logs = subprocess.run(["docker", "logs", self.container_name], capture_output=True, text=True)
            if "Remote interface available" in logs.stdout or "Started" in logs.stdout:
                console.print("✓ Neo4j (docker) is ready")
                return
            time.sleep(1)
        console.print("[yellow]Proceeding without confirmed readiness; service should be up shortly[/yellow]")
    
    def _validate_csv_files(self) -> List[Path]:
        """Validate that required CSV files exist."""
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {self.csv_dir}")
        
        csv_files = list(self.csv_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_dir}")
        
        console.print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            console.print(f"  ✓ {csv_file.name}")
        
        return csv_files
    
    def _stop_neo4j(self):
        """Stop Neo4j server."""
        console.print("Stopping Neo4j server...")
        try:
            result = subprocess.run([str(self.neo4j_ctl), "stop"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0 and "not running" not in result.stdout:
                console.print(f"[yellow]Warning: {result.stdout}[/yellow]")
        except subprocess.TimeoutExpired:
            console.print("[yellow]Neo4j stop timed out, proceeding anyway[/yellow]")
        
        # Wait a moment for complete shutdown
        time.sleep(2)
    
    def _start_neo4j(self):
        """Start Neo4j server."""
        console.print("Starting Neo4j server...")
        result = subprocess.run([str(self.neo4j_ctl), "start"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Neo4j: {result.stderr}")
        
        # Wait for Neo4j to be ready
        console.print("Waiting for Neo4j to be ready...")
        max_wait = 30
        for i in range(max_wait):
            try:
                result = subprocess.run([str(self.neo4j_ctl), "status"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "running" in result.stdout:
                    console.print("✓ Neo4j is ready")
                    return
            except:
                pass
            time.sleep(1)
        
        raise RuntimeError("Neo4j failed to start within 30 seconds")
    
    def _prepare_database(self):
        """Prepare database for bulk import."""
        # For neo4j-admin import, we need to delete the existing database
        db_dir = self.neo4j_home / "data" / "databases" / self.database_name
        if db_dir.exists():
            console.print(f"Removing existing database: {db_dir}")
            shutil.rmtree(db_dir)
    
    def _run_bulk_import(self, csv_files: List[Path]) -> Dict[str, Any]:
        """Run neo4j-admin database import command."""
        # Separate node and relationship files
        node_files = []
        rel_files = []
        
        for csv_file in csv_files:
            if "relationships" in csv_file.name:
                rel_files.append(csv_file)
            else:
                node_files.append(csv_file)
        
        # Build import command
        cmd = [
            str(self.neo4j_admin), 
            "database", "import", "full",
            "--overwrite-destination=true",
        ]
        
        # Add node files with labels
        for node_file in node_files:
            # Extract label from filename with explicit overrides where needed.
            stem = node_file.stem.lower()
            label_overrides = {
                # Canonical node labels
                "domainannotations": "DomainAnnotation",
                "functionalannotations": "FunctionalAnnotation",
                "keggorthologs": "KEGGOrtholog",
                "qualitymetrics": "QualityMetrics",
                # BGC variants
                "bgcs": "Bgc",
                "bgc_clusters": "Bgc",
                # CAZy
                "cazymeannotations": "Cazymeannotation",
                "cazymefamilies": "Cazymefamily",
            }
            if stem in label_overrides:
                label = label_overrides[stem]
            else:
                # Default heuristic (e.g., "domains" -> "Domain")
                label = node_file.stem.rstrip('s').title()
                if label.endswith('ie'):  # Fix "Qualitymetrie" -> "QualityMetrics"
                    label = label[:-2] + "ies"
                elif label == "Domainannotation":
                    label = "DomainAnnotation"
                elif label == "Functionalannotation":
                    label = "FunctionalAnnotation"
                elif label == "Keggortholog":
                    label = "KEGGOrtholog"
                elif label == "Qualitymetric":
                    label = "QualityMetrics"

            cmd.extend(["--nodes", f"{label}={node_file}"])
        
        # Add relationship files  
        for rel_file in rel_files:
            # Extract relationship type from filename
            rel_type = rel_file.stem.replace("_relationships", "").upper()
            cmd.extend(["--relationships", f"{rel_type}={rel_file}"])
                
        console.print(f"Running import command...")
        console.print(f"Command: {' '.join(cmd)}")
        
        # Run import
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            console.print(f"[red]Import failed![/red]")
            console.print(f"STDOUT: {result.stdout}")
            console.print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"neo4j-admin import failed: {result.stderr}")
        
        console.print(f"✓ Import successful!")
        console.print(f"Import output: {result.stdout}")
        
        return {"import_output": result.stdout}
    
    def _create_constraints_and_indexes(self):
        """Create constraints and indexes after import (enabled by default, idempotent).

        Behavior:
        - Attempts to connect using environment vars when provided.
        - Falls back to unauthenticated local connection (bolt://localhost:7687) when auth is disabled (e.g., docker engine uses NEO4J_AUTH=none).
        - Silently continues if the driver is unavailable or the connection fails.
        """
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        try:
            from neo4j import GraphDatabase
        except Exception as e:
            console.print(f"[yellow]Neo4j driver not available; skipping constraints/indexes: {e}[/yellow]")
            return

        # Choose auth mode: explicit creds if both provided; otherwise no-auth
        auth = (user, password) if (user and password) else None
        try:
            driver = GraphDatabase.driver(uri, auth=auth)
            # Quick ping
            with driver.session() as _s:
                _s.run("RETURN 1 AS ok").single()
        except Exception as e:
            console.print(f"[yellow]Skipping constraints/indexes (connection failed): {e}[/yellow]")
            return

        statements = [
                # Uniqueness constraints on core IDs
                "CREATE CONSTRAINT genome_id IF NOT EXISTS FOR (g:Genome) REQUIRE g.id IS UNIQUE",
                "CREATE CONSTRAINT genome_genomeId IF NOT EXISTS FOR (g:Genome) REQUIRE g.genomeId IS UNIQUE",
                "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE",
                "CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT domain_id IF NOT EXISTS FOR (d:Domain) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT domain_annotation_id IF NOT EXISTS FOR (da:DomainAnnotation) REQUIRE da.id IS UNIQUE",
                "CREATE CONSTRAINT kegg_id IF NOT EXISTS FOR (k:KEGGOrtholog) REQUIRE k.id IS UNIQUE",
                "CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE",
                "CREATE CONSTRAINT bgc_id IF NOT EXISTS FOR (b:Bgc) REQUIRE b.id IS UNIQUE",
                # CAZy labels (post-import)
                "CREATE CONSTRAINT cazy_family_id IF NOT EXISTS FOR (f:Cazymefamily) REQUIRE f.familyId IS UNIQUE",
                "CREATE CONSTRAINT cazy_ann_id IF NOT EXISTS FOR (a:Cazymeannotation) REQUIRE a.annotationId IS UNIQUE",
                # Composite index for spatial gene scans
                "CREATE INDEX gene_contig_coords IF NOT EXISTS FOR (g:Gene) ON (g.contig, g.startCoordinate, g.endCoordinate)",
                "CREATE INDEX gene_contig_start IF NOT EXISTS FOR (g:Gene) ON (g.contig, g.startCoordinate)",
                # CRISPR array spatial indexes
                "CREATE INDEX crispr_contig_coords IF NOT EXISTS FOR (ca:CrisprArray) ON (ca.contig, ca.startCoordinate, ca.endCoordinate)",
                "CREATE INDEX crispr_contig_start IF NOT EXISTS FOR (ca:CrisprArray) ON (ca.contig, ca.startCoordinate)",
            # Helpful single-property indexes
            "CREATE INDEX protein_name IF NOT EXISTS FOR (p:Protein) ON (p.name)",
            "CREATE INDEX domain_name IF NOT EXISTS FOR (d:Domain) ON (d.name)",
            "CREATE INDEX domain_pfamAccession IF NOT EXISTS FOR (d:Domain) ON (d.pfamAccession)",
            "CREATE INDEX kegg_desc IF NOT EXISTS FOR (k:KEGGOrtholog) ON (k.description)",
            # Full-text indexes (optional)
            "CREATE FULLTEXT INDEX proteinText IF NOT EXISTS FOR (p:Protein) ON EACH [p.name, p.description]",
            "CREATE FULLTEXT INDEX domainText IF NOT EXISTS FOR (d:Domain) ON EACH [d.id, d.name, d.description]",
            "CREATE FULLTEXT INDEX keggText IF NOT EXISTS FOR (k:KEGGOrtholog) ON EACH [k.id, k.description]",
            "CREATE FULLTEXT INDEX pathwayText IF NOT EXISTS FOR (pw:Pathway) ON EACH [pw.id, pw.name, pw.description]",
        ]

        with driver.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception as e:
                    # Most errors are benign (exists) due to IF NOT EXISTS
                    if "already exists" not in str(e):
                        console.print(f"[yellow]Index/constraint warning: {e}[/yellow]")

        driver.close()

    def _precompute_neighbor_edges(self, batch_size: int = 2000):
        """Precompute genomic NEXT edges efficiently by streaming per contig.

        Performance-oriented approach:
        - Only create :NEXT edges (use incoming :NEXT for previous traversal)
        - Set properties: {contig, delta, same_strand}
        - Stream genes per contig ordered by startCoordinate, compute pairs client-side
        - UNWIND pairs in batches to MERGE relationships
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            console.print("[yellow]Warning: neo4j driver not available, skipping neighbor precompute[/yellow]")
            return

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Queries
        list_contigs = (
            "MATCH (g:Gene) WHERE g.contig IS NOT NULL RETURN DISTINCT g.contig AS contig ORDER BY contig"
        )
        genes_for_contig = (
            "MATCH (g:Gene {contig: $contig}) "
            "WHERE g.startCoordinate IS NOT NULL AND g.endCoordinate IS NOT NULL "
            "RETURN g.id AS id, g.startCoordinate AS start, g.endCoordinate AS end, g.strand AS strand "
            "ORDER BY g.startCoordinate"
        )
        create_next_batch = (
            "UNWIND $pairs AS row "
            "MATCH (a:Gene {id: row.a}) MATCH (b:Gene {id: row.b}) "
            "MERGE (a)-[r:NEXT]->(b) "
            "SET r.contig = $contig, r.delta = row.delta, r.same_strand = row.same_strand"
        )

        total_pairs = 0
        with driver.session() as session:
            # Get contigs
            contigs = [rec["contig"] for rec in session.run(list_contigs)]
            console.print(f"Found {len(contigs)} contigs to link with NEXT edges")

            for idx, contig in enumerate(contigs, 1):
                # Fetch genes ordered by start
                records = list(session.run(genes_for_contig, contig=contig))
                n = len(records)
                if n < 2:
                    continue

                # Build adjacency pairs for this contig
                pairs = []
                try:
                    for i in range(n - 1):
                        a = records[i]
                        b = records[i + 1]
                        a_end = int(a["end"]) if a["end"] is not None else 0
                        b_start = int(b["start"]) if b["start"] is not None else a_end
                        same_strand = str(a["strand"]) == str(b["strand"]) if a["strand"] is not None and b["strand"] is not None else False
                        pairs.append({
                            "a": a["id"],
                            "b": b["id"],
                            "delta": int(b_start) - int(a_end),
                            "same_strand": bool(same_strand),
                        })
                except Exception as e:
                    console.print(f"[yellow]Skipping contig {contig} due to data error: {e}[/yellow]")
                    continue

                total_pairs += len(pairs)

                # Write in batches
                for j in range(0, len(pairs), batch_size):
                    chunk = pairs[j:j + batch_size]
                    session.run(create_next_batch, pairs=chunk, contig=contig)

                if idx % 50 == 0:
                    console.print(f"  Linked {idx}/{len(contigs)} contigs so far...")

        driver.close()
        console.print(f"[green]✓ Created/merged ~{total_pairs:,} NEXT edges[/green]")


def main():
    """Main execution function."""
    csv_dir = Path("data/stage07_kg/csv")
    
    if not csv_dir.exists():
        console.print(f"[red]CSV directory not found: {csv_dir}[/red]")
        console.print("Run rdf_to_csv.py first to generate CSV files")
        return 1
    
    try:
        loader = Neo4jBulkLoader(csv_dir)
        stats = loader.bulk_import()
        
        console.print(f"\n[bold green]✓ Bulk import completed successfully![/bold green]")
        console.print(f"Import time: {stats['import_time_seconds']:.2f} seconds")
        console.print(f"CSV files processed: {stats['csv_files_processed']}")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Bulk import failed: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
