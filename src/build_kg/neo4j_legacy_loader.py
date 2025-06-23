#!/usr/bin/env python3
"""
Load RDF knowledge graph into Neo4j database.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import time

from neo4j import GraphDatabase
import rdflib
from rich.console import Console
from rich.progress import Progress

console = Console()
logger = logging.getLogger(__name__)


class Neo4jLoader:
    """Load RDF triples into Neo4j graph database."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "your_new_password", database: str = "neo4j"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        console.print(f"Connected to Neo4j at {uri} (database: {database})")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            console.print("[yellow]Clearing existing database...[/yellow]")
            session.run("MATCH (n) DETACH DELETE n")
            console.print("✓ Database cleared")
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        constraints = [
            "CREATE CONSTRAINT genome_id IF NOT EXISTS FOR (g:Genome) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE", 
            "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT domain_annotation_id IF NOT EXISTS FOR (d:DomainAnnotation) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT domain_id IF NOT EXISTS FOR (f:Domain) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT kegg_id IF NOT EXISTS FOR (k:KEGGOrtholog) REQUIRE k.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            console.print("Creating constraints and indexes...")
            for constraint in constraints:
                try:
                    session.run(constraint)
                    console.print(f"✓ {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    if "already exists" not in str(e):
                        console.print(f"⚠️  {e}")
    
    def load_csv_data(self, csv_dir: Path) -> Dict[str, Any]:
        """Load knowledge graph from CSV files with critical relationships."""
        import csv
        from pathlib import Path
        
        console.print(f"[bold blue]Loading knowledge graph from CSV: {csv_dir}[/bold blue]")
        
        def load_nodes_batch(session, csv_file, label, batch_size=1000):
            console.print(f'Loading {label} nodes from {csv_file.name}...')
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                nodes = []
                
                for row in reader:
                    # Prepare node properties
                    props = {}
                    for key, value in row.items():
                        if key and key != 'id:ID' and value:  # Skip None keys, ID, and empty values
                            props[key] = value
                    
                    nodes.append({
                        'id': row['id:ID'],
                        'properties': props
                    })
                    
                    # Process batch
                    if len(nodes) >= batch_size:
                        create_batch = f'''
                        UNWIND $nodes as node
                        CREATE (n:{label} {{id: node.id}})
                        SET n += node.properties
                        '''
                        session.run(create_batch, nodes=nodes)
                        nodes = []
                
                # Process remaining nodes
                if nodes:
                    create_batch = f'''
                    UNWIND $nodes as node
                    CREATE (n:{label} {{id: node.id}})
                    SET n += node.properties
                    '''
                    session.run(create_batch, nodes=nodes)
            
            console.print(f'✓ Loaded {label} nodes')

        def load_relationships_batch(session, csv_file, rel_type, batch_size=500):
            console.print(f'Loading {rel_type} relationships from {csv_file.name}...')
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rels = []
                
                for row in reader:
                    rels.append({
                        'start_id': row[':START_ID'],
                        'end_id': row[':END_ID']
                    })
                    
                    # Process batch with optimized query
                    if len(rels) >= batch_size:
                        create_batch = f'''
                        UNWIND $rels as rel
                        MERGE (a {{id: rel.start_id}})
                        MERGE (b {{id: rel.end_id}})
                        MERGE (a)-[:{rel_type}]->(b)
                        '''
                        session.run(create_batch, rels=rels)
                        rels = []
                
                # Process remaining relationships
                if rels:
                    create_batch = f'''
                    UNWIND $rels as rel
                    MERGE (a {{id: rel.start_id}})
                    MERGE (b {{id: rel.end_id}})
                    MERGE (a)-[:{rel_type}]->(b)
                    '''
                    session.run(create_batch, rels=rels)
            
            console.print(f'✓ Loaded {rel_type} relationships')
        
        with self.driver.session() as session:
            # Load core nodes
            load_nodes_batch(session, csv_dir / 'pathways.csv', 'Pathway')
            load_nodes_batch(session, csv_dir / 'keggorthologs.csv', 'KEGGOrtholog') 
            load_nodes_batch(session, csv_dir / 'proteins.csv', 'Protein')
            load_nodes_batch(session, csv_dir / 'genes.csv', 'Gene')
            load_nodes_batch(session, csv_dir / 'domains.csv', 'Domain')
            load_nodes_batch(session, csv_dir / 'domainannotations.csv', 'DomainAnnotation')
            
            # Load critical relationships for ribosomal protein queries
            load_relationships_batch(session, csv_dir / 'hasfunction_relationships.csv', 'HASFUNCTION')
            load_relationships_batch(session, csv_dir / 'encodedby_relationships.csv', 'ENCODEDBY')
            load_relationships_batch(session, csv_dir / 'participatesin_relationships.csv', 'PARTICIPATESIN')
            load_relationships_batch(session, csv_dir / 'hasdomain_relationships.csv', 'HASDOMAIN')
            load_relationships_batch(session, csv_dir / 'domainfamily_relationships.csv', 'DOMAINFAMILY')
            
            console.print('[green]✓ Critical knowledge graph data loaded![/green]')
            
            # Test queries
            console.print('\n[blue]Testing database...[/blue]')
            
            result = session.run('MATCH (p:Pathway) RETURN count(p) as count')
            pathway_count = result.single()['count']
            console.print(f'Pathways loaded: {pathway_count}')
            
            result = session.run("MATCH (ko:KEGGOrtholog) WHERE toLower(ko.description) CONTAINS 'ribosom' RETURN count(ko) as count")
            ribo_count = result.single()['count']
            console.print(f'Ribosomal KOs found: {ribo_count}')
            
            result = session.run('MATCH ()-[r:HASFUNCTION]->() RETURN count(r) as count')
            hasfunction_count = result.single()['count']
            console.print(f'HASFUNCTION relationships: {hasfunction_count}')
            
            result = session.run('MATCH ()-[r:ENCODEDBY]->() RETURN count(r) as count')
            encodedby_count = result.single()['count']
            console.print(f'ENCODEDBY relationships: {encodedby_count}')
        
        return {
            'pathways': pathway_count,
            'ribosomal_kos': ribo_count,
            'hasfunction_rels': hasfunction_count,
            'encodedby_rels': encodedby_count
        }

    def load_csv_data_bulk(self, csv_dir: Path) -> Dict[str, Any]:
        """Load knowledge graph using fast neo4j-admin bulk import."""
        import subprocess
        import tempfile
        import shutil
        
        console.print(f"[bold blue]Loading knowledge graph using neo4j-admin bulk import: {csv_dir}[/bold blue]")
        
        # Check if Neo4j is running and stop it
        console.print("Stopping Neo4j service for bulk import...")
        try:
            subprocess.run(['sudo', 'systemctl', 'stop', 'neo4j'], check=True, capture_output=True)
            console.print("✓ Neo4j service stopped")
        except subprocess.CalledProcessError as e:
            console.print(f"⚠️  Could not stop Neo4j service: {e}")
            console.print("Trying to stop using neo4j command...")
            try:
                subprocess.run(['neo4j', 'stop'], check=True, capture_output=True)
                console.print("✓ Neo4j stopped")
            except subprocess.CalledProcessError:
                console.print("[red]Warning: Could not stop Neo4j. Bulk import may fail.[/red]")
        
        # Backup existing database
        neo4j_data_dir = Path("/var/lib/neo4j/data")
        if neo4j_data_dir.exists():
            backup_dir = Path(f"/var/lib/neo4j/data_backup_{int(time.time())}")
            console.print(f"Backing up existing database to {backup_dir}")
            try:
                shutil.move(str(neo4j_data_dir), str(backup_dir))
                console.print("✓ Database backed up")
            except Exception as e:
                console.print(f"⚠️  Could not backup database: {e}")
        
        # Build neo4j-admin import command (use default database name)
        import_cmd = [
            'neo4j-admin', 'database', 'import', 'full',
            '--overwrite-destination=true',
            '--verbose',
            '--ignore-extra-columns=true',
            '--skip-bad-relationships=true',
            '--bad-tolerance=20000',  # Allow skipping bad relationships
            # Node files
            f'--nodes=Pathway={csv_dir}/pathways.csv',
            f'--nodes=KEGGOrtholog={csv_dir}/keggorthologs.csv', 
            f'--nodes=Protein={csv_dir}/proteins.csv',
            f'--nodes=Gene={csv_dir}/genes.csv',
            f'--nodes=Domain={csv_dir}/domains.csv',
            f'--nodes=DomainAnnotation={csv_dir}/domainannotations.csv',
            f'--nodes=FunctionalAnnotation={csv_dir}/functionalannotations.csv',
            f'--nodes=QualityMetrics={csv_dir}/qualitymetricss.csv',
            f'--nodes=Genome={csv_dir}/genomes.csv',
            # Relationship files (excluding problematic pathway relationships)
            f'--relationships=HASFUNCTION={csv_dir}/hasfunction_relationships.csv',
            f'--relationships=ENCODEDBY={csv_dir}/encodedby_relationships.csv',
            # f'--relationships=PARTICIPATESIN={csv_dir}/participatesin_relationships.csv',  # Skip for now
            f'--relationships=HASDOMAIN={csv_dir}/hasdomain_relationships.csv',
            f'--relationships=DOMAINFAMILY={csv_dir}/domainfamily_relationships.csv',
            f'--relationships=BELONGSTOGENOME={csv_dir}/belongstogenome_relationships.csv',
            f'--relationships=BELONGSTOPROTEIN={csv_dir}/belongstoprotein_relationships.csv',
            f'--relationships=ANNOTATESPROTEIN={csv_dir}/annotatesprotein_relationships.csv',
            f'--relationships=ASSIGNEDFUNCTION={csv_dir}/assignedfunction_relationships.csv',
            f'--relationships=HASQUALITYMETRICS={csv_dir}/hasqualitymetrics_relationships.csv'
            # No database name - uses default 'neo4j'
        ]
        
        console.print("Running neo4j-admin bulk import...")
        console.print(f"Command: {' '.join(import_cmd)}")
        
        try:
            result = subprocess.run(import_cmd, capture_output=True, text=True, check=True)
            console.print("✓ Bulk import completed successfully!")
            if result.stdout:
                console.print(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Bulk import failed: {e}[/red]")
            console.print(f"Error output: {e.stderr}")
            raise
        
        # Start Neo4j service
        console.print("Starting Neo4j service...")
        try:
            subprocess.run(['sudo', 'systemctl', 'start', 'neo4j'], check=True, capture_output=True)
            console.print("✓ Neo4j service started")
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['neo4j', 'start'], check=True, capture_output=True)
                console.print("✓ Neo4j started")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Could not start Neo4j: {e}[/red]")
                raise
        
        # Wait for Neo4j to be ready
        console.print("Waiting for Neo4j to be ready...")
        time.sleep(10)
        
        # Test the imported database
        console.print('\n[blue]Testing imported database...[/blue]')
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run('MATCH (p:Pathway) RETURN count(p) as count')
                pathway_count = result.single()['count']
                console.print(f'Pathways loaded: {pathway_count}')
                
                result = session.run("MATCH (ko:KEGGOrtholog) WHERE toLower(ko.description) CONTAINS 'ribosom' RETURN count(ko) as count")
                ribo_count = result.single()['count']
                console.print(f'Ribosomal KOs found: {ribo_count}')
                
                result = session.run('MATCH ()-[r:HASFUNCTION]->() RETURN count(r) as count')
                hasfunction_count = result.single()['count']
                console.print(f'HASFUNCTION relationships: {hasfunction_count}')
                
                result = session.run('MATCH ()-[r:ENCODEDBY]->() RETURN count(r) as count')
                encodedby_count = result.single()['count']
                console.print(f'ENCODEDBY relationships: {encodedby_count}')
                
                result = session.run('MATCH ()-[r:PARTICIPATESIN]->() RETURN count(r) as count')
                participatesin_count = result.single()['count'] 
                console.print(f'PARTICIPATESIN relationships: {participatesin_count}')
                
                console.print('[green]✓ Database import successful![/green]')
                
                return {
                    'pathways': pathway_count,
                    'ribosomal_kos': ribo_count,
                    'hasfunction_rels': hasfunction_count,
                    'encodedby_rels': encodedby_count,
                    'participatesin_rels': participatesin_count
                }
        except Exception as e:
            console.print(f"[red]Error testing database: {e}[/red]")
            raise

    def load_rdf_graph(self, rdf_file: Path) -> Dict[str, Any]:
        """Load RDF triples into Neo4j."""
        console.print(f"[bold blue]Loading RDF knowledge graph: {rdf_file}[/bold blue]")
        
        # Load RDF graph
        g = rdflib.Graph()
        g.parse(rdf_file, format="turtle")
        console.print(f"Loaded {len(g)} RDF triples")
        
        # Convert RDF to Neo4j
        stats = self._convert_rdf_to_neo4j(g)
        
        return stats
    
    def _convert_rdf_to_neo4j(self, g: rdflib.Graph) -> Dict[str, Any]:
        """Convert RDF triples to Neo4j nodes and relationships with proper batching."""
        
        # Parse namespaces for cleaner URIs
        namespaces = {
            "genome": "http://genome-kg.org/genomes/",
            "protein": "http://genome-kg.org/proteins/", 
            "gene": "http://genome-kg.org/genes/",
            "pfam": "http://pfam.xfam.org/family/",
            "ko": "http://www.genome.jp/kegg/ko/",
            "kg": "http://genome-kg.org/ontology/"
        }
        
        # First pass: collect RDF types and group triples by subject
        rdf_types = {}
        entity_properties = {}  # subject -> {property: value}
        relationships = []  # [(subj, pred, obj)]
        
        for subj, pred, obj in g:
            subj_str = str(subj)
            pred_str = str(pred)
            
            if pred_str == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                rdf_types[subj_str] = str(obj)
            elif isinstance(obj, rdflib.URIRef):
                # This is a relationship
                relationships.append((subj, pred, obj))
            else:
                # This is a property
                if subj_str not in entity_properties:
                    entity_properties[subj_str] = {}
                
                pred_name = self._uri_to_property(pred_str, namespaces)
                
                # Handle different data types
                if isinstance(obj, rdflib.Literal):
                    if obj.datatype == rdflib.XSD.decimal or obj.datatype == rdflib.XSD.double:
                        obj_value = float(obj)
                    elif obj.datatype == rdflib.XSD.integer:
                        obj_value = int(obj)
                    else:
                        obj_value = str(obj)
                else:
                    obj_value = str(obj)
                
                entity_properties[subj_str][pred_name] = obj_value
        
        nodes_created = 0
        relationships_created = 0
        
        with self.driver.session() as session:
            with Progress(console=console) as progress:
                # Step 1: Create all nodes with their properties in batches
                node_task = progress.add_task("Creating nodes...", total=len(entity_properties))
                
                # Process nodes in batches of 100
                batch_size = 100
                entity_items = list(entity_properties.items())
                
                for i in range(0, len(entity_items), batch_size):
                    batch = entity_items[i:i + batch_size]
                    
                    # Build batch Cypher query
                    batch_cypher_parts = []
                    batch_params = {}
                    
                    for j, (subj_uri, props) in enumerate(batch):
                        subj_id = self._uri_to_id(subj_uri, namespaces)
                        node_type = self._get_node_type_from_rdf(subj_uri, rdf_types, namespaces)
                        
                        # Create parameter names for this node
                        id_param = f"id_{j}"
                        batch_params[id_param] = subj_id
                        
                        # Build SET clause for properties
                        set_clauses = []
                        for prop_name, prop_value in props.items():
                            prop_param = f"prop_{j}_{prop_name}"
                            # Replace invalid characters in parameter names
                            prop_param = prop_param.replace('-', '_').replace('.', '_')
                            batch_params[prop_param] = prop_value
                            set_clauses.append(f"n_{j}.{prop_name} = ${prop_param}")
                        
                        set_clause = ", ".join(set_clauses) if set_clauses else ""
                        
                        # Add to batch
                        node_clause = f"MERGE (n_{j}:{node_type} {{id: ${id_param}}})"
                        if set_clause:
                            node_clause += f" SET {set_clause}"
                        
                        batch_cypher_parts.append(node_clause)
                    
                    # Execute batch
                    if batch_cypher_parts:
                        batch_cypher = "\n".join(batch_cypher_parts)
                        session.run(batch_cypher, **batch_params)
                        nodes_created += len(batch)
                    
                    progress.advance(node_task, len(batch))
                
                # Step 2: Create all relationships in batches
                rel_task = progress.add_task("Creating relationships...", total=len(relationships))
                
                for i in range(0, len(relationships), batch_size):
                    batch = relationships[i:i + batch_size]
                    
                    batch_cypher_parts = []
                    batch_params = {}
                    
                    for j, (subj, pred, obj) in enumerate(batch):
                        subj_id = self._uri_to_id(str(subj), namespaces)
                        obj_id = self._uri_to_id(str(obj), namespaces)
                        pred_name = self._uri_to_property(str(pred), namespaces)
                        
                        # Determine node types
                        subj_type = self._get_node_type_from_rdf(str(subj), rdf_types, namespaces)
                        obj_type = self._get_node_type_from_rdf(str(obj), rdf_types, namespaces)
                        
                        # Create parameter names
                        subj_param = f"subj_{j}"
                        obj_param = f"obj_{j}"
                        batch_params[subj_param] = subj_id
                        batch_params[obj_param] = obj_id
                        
                        # Add relationship
                        rel_clause = f"""
                        MERGE (s_{j}:{subj_type} {{id: ${subj_param}}})
                        MERGE (o_{j}:{obj_type} {{id: ${obj_param}}})
                        MERGE (s_{j})-[:{pred_name}]->(o_{j})
                        """
                        batch_cypher_parts.append(rel_clause)
                    
                    # Execute batch
                    if batch_cypher_parts:
                        batch_cypher = "\n".join(batch_cypher_parts)
                        session.run(batch_cypher, **batch_params)
                        relationships_created += len(batch)
                    
                    progress.advance(rel_task, len(batch))
        
        console.print(f"✓ Created {nodes_created} nodes with properties")
        console.print(f"✓ Created {relationships_created} relationships")
        
        return {
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "total_triples": len(g)
        }
    
    def _uri_to_id(self, uri: str, namespaces: Dict[str, str]) -> str:
        """Convert URI to readable ID."""
        for prefix, namespace in namespaces.items():
            if uri.startswith(namespace):
                return uri.replace(namespace, "")
        return uri.split("/")[-1]  # Fallback
    
    def _uri_to_property(self, uri: str, namespaces: Dict[str, str]) -> str:
        """Convert property URI to readable name."""
        if uri.startswith(namespaces["kg"]):
            return uri.replace(namespaces["kg"], "")
        return uri.split("/")[-1].split("#")[-1]
    
    def _get_node_type_from_rdf(self, uri: str, rdf_types: Dict[str, str], namespaces: Dict[str, str]) -> str:
        """Determine Neo4j node type from RDF type information."""
        # First check RDF type declarations
        if uri in rdf_types:
            rdf_type = rdf_types[uri]
            if rdf_type.endswith("Genome"):
                return "Genome"
            elif rdf_type.endswith("Protein"):
                return "Protein"
            elif rdf_type.endswith("Gene"):
                return "Gene"
            elif rdf_type.endswith("Domain"):
                return "Domain"
            elif rdf_type.endswith("KEGGOrtholog"):
                return "KEGGOrtholog"
            elif rdf_type.endswith("QualityMetrics"):
                return "QualityMetrics"
            elif rdf_type.endswith("DomainAnnotation"):
                return "DomainAnnotation"
            elif rdf_type.endswith("FunctionalAnnotation"):
                return "FunctionalAnnotation"
        
        # Fallback to URI pattern matching
        return self._get_node_type(uri, namespaces)
    
    def _get_node_type(self, uri: str, namespaces: Dict[str, str]) -> str:
        """Determine Neo4j node type from URI (fallback method)."""
        if uri.startswith(namespaces["genome"]):
            # Check if it's a quality metrics node
            if "/quality" in uri:
                return "QualityMetrics"
            return "Genome"
        elif uri.startswith(namespaces["protein"]):
            return "Protein"
        elif uri.startswith(namespaces["gene"]):
            return "Gene"
        elif uri.startswith(namespaces["pfam"]):
            return "Domain"
        elif uri.startswith(namespaces["ko"]):
            return "KEGGOrtholog"
        else:
            # Default fallback
            return "Node"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            for record in result:
                label = record["labels"][0] if record["labels"] else "Unknown"
                stats[f"{label}_nodes"] = record["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            for record in result:
                stats[f"{record['rel_type']}_relationships"] = record["count"]
            
            # Total counts
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            stats["total_nodes"] = total_nodes
            stats["total_relationships"] = total_rels
            
        return stats


def main():
    """Main execution function - load from corrected CSV files."""
    console.print("[bold green]Neo4j Knowledge Graph Loader[/bold green]")
    
    # Use corrected CSV files
    csv_dir = Path("data/stage05_kg_corrected/csv")
    
    if not csv_dir.exists():
        console.print(f"[red]CSV directory not found: {csv_dir}[/red]")
        console.print("Run RDF builder first to generate corrected CSV files")
        return
    
    # Load into Neo4j
    loader = Neo4jLoader()
    
    try:
        # Use fast bulk import with all required nodes
        stats = loader.load_csv_data_bulk(csv_dir)
        
        console.print(f"\n[bold green]✓ Knowledge graph loaded successfully![/bold green]")
        console.print(f"Pathways: {stats['pathways']}")
        console.print(f"Ribosomal KOs: {stats['ribosomal_kos']}")
        console.print(f"HASFUNCTION relationships: {stats['hasfunction_rels']}")
        console.print(f"ENCODEDBY relationships: {stats['encodedby_rels']}")
        
    except Exception as e:
        console.print(f"[red]Error loading knowledge graph: {e}[/red]")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()