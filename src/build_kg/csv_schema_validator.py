#!/usr/bin/env python3
"""
Fast CSV Schema Validator for Neo4j Bulk Import

Validates all CSV files against expected schema before running expensive Neo4j import.
Should catch 99%+ of import failures in <5 seconds vs 10+ minutes for bulk import.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Set, Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import sys

@dataclass
class ValidationError:
    file: str
    line: Optional[int]
    field: Optional[str] 
    error: str
    severity: str  # 'ERROR', 'WARNING'

@dataclass
class NodeSchema:
    label: str
    filename: str
    required_headers: List[str]
    optional_headers: List[str] = None
    id_prefix: str = ""

@dataclass
class RelationshipSchema:
    relationship_type: str
    filename: str
    start_node_prefixes: List[str]  # Valid prefixes for START_ID
    end_node_prefixes: List[str]    # Valid prefixes for END_ID

class CSVSchemaValidator:
    """Fast comprehensive CSV validation for Neo4j import."""
    
    def __init__(self, csv_dir: Path):
        self.csv_dir = csv_dir
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.node_ids: Dict[str, Set[str]] = defaultdict(set)
        self.all_node_ids: Set[str] = set()
        
        # Define expected schema
        self.node_schemas = [
            NodeSchema("Genome", "genomes.csv", ["id:ID", "genomeId"]),
            NodeSchema("Gene", "genes.csv", ["id:ID", "geneId", "startCoordinate", "endCoordinate"]),
            NodeSchema("Protein", "proteins.csv", ["id:ID", "proteinId", "length"]),
            NodeSchema("FunctionalAnnotation", "functionalannotations.csv", ["id:ID", "bitscore", "confidence", "evalue"]),
            NodeSchema("DomainAnnotation", "domainannotations.csv", ["id:ID", "bitscore", "domainEnd", "domainStart", "evalue"]),
            NodeSchema("Domain", "domains.csv", ["id:ID", "pfamAccession", "description", "familyType"]),
            NodeSchema("KEGGOrtholog", "keggorthologs.csv", ["id:ID", "koId", "description"]),
            NodeSchema("Pathway", "pathways.csv", ["id:ID", "pathwayNumber", "description"]),
            NodeSchema("Contig", "contigs.csv", ["id:ID", "contigId"]),
            NodeSchema("QualityMetrics", "qualitymetrics.csv", ["id:ID"]),
            NodeSchema("Dataset", "datasets.csv", ["id:ID"]),
            NodeSchema("Entity", "entitys.csv", ["id:ID"]),
        ]
        
        self.relationship_schemas = [
            RelationshipSchema("ENCODEDBY", "encodedby_relationships.csv", ["protein"], ["gene"]),
            RelationshipSchema("BELONGSTOGENOME", "belongstogenome_relationships.csv", ["gene"], [""]),  # genome has no prefix
            RelationshipSchema("BELONGSTOCONTIG", "belongstocontig_relationships.csv", ["gene"], ["contig"]),
            RelationshipSchema("BELONGSTOPROTEIN", "belongstoprotein_relationships.csv", ["protein"], ["protein"]),  # domain annotation -> protein
            RelationshipSchema("DOMAINFAMILY", "domainfamily_relationships.csv", ["protein"], [""]),  # domain annotation -> pfam (no prefix)
            RelationshipSchema("ANNOTATESPROTEIN", "annotatesprotein_relationships.csv", ["protein"], ["protein"]),  # func annotation -> protein
            RelationshipSchema("ASSIGNEDFUNCTION", "assignedfunction_relationships.csv", ["protein"], [""]),  # func annotation -> kegg (no prefix)
            RelationshipSchema("HASDOMAIN", "hasdomain_relationships.csv", ["protein"], ["protein"]),  # protein -> domain annotation
            RelationshipSchema("HASFUNCTION", "hasfunction_relationships.csv", ["protein"], [""]),  # protein -> kegg (no prefix)
            RelationshipSchema("HASPARTICIPANT", "hasparticipant_relationships.csv", ["pathway"], [""]),  # pathway -> kegg
            RelationshipSchema("PARTICIPATESIN", "participatesin_relationships.csv", [""], ["pathway"]),  # kegg -> pathway
            RelationshipSchema("HASQUALITYMETRICS", "hasqualitymetrics_relationships.csv", [""], ["genome"]),  # genome -> quality metrics
        ]
    
    def add_error(self, file: str, error: str, line: int = None, field: str = None):
        """Add validation error."""
        self.errors.append(ValidationError(file, line, field, error, "ERROR"))
    
    def add_warning(self, file: str, warning: str, line: int = None, field: str = None):
        """Add validation warning."""
        self.warnings.append(ValidationError(file, warning, line, field, "WARNING"))
    
    def validate_file_exists(self, filename: str) -> bool:
        """Check if required file exists."""
        filepath = self.csv_dir / filename
        if not filepath.exists():
            self.add_error(filename, f"Missing required file: {filepath}")
            return False
        return True
    
    def validate_csv_structure(self, schema: NodeSchema) -> bool:
        """Validate CSV file structure and headers."""
        if not self.validate_file_exists(schema.filename):
            return False
            
        filepath = self.csv_dir / schema.filename
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                # Check if file is empty
                content = csvfile.read().strip()
                if not content:
                    self.add_error(schema.filename, "File is empty")
                    return False
                
                csvfile.seek(0)
                reader = csv.reader(csvfile)
                
                try:
                    headers = next(reader)
                except StopIteration:
                    self.add_error(schema.filename, "No headers found")
                    return False
                
                # Check required headers
                missing_headers = []
                for required in schema.required_headers:
                    if required not in headers:
                        missing_headers.append(required)
                
                if missing_headers:
                    self.add_error(schema.filename, f"Missing required headers: {missing_headers}")
                    return False
                
                # Check for empty required columns
                if 'id:ID' not in headers:
                    self.add_error(schema.filename, "Missing id:ID column")
                    return False
                    
                return True
                
        except Exception as e:
            self.add_error(schema.filename, f"Failed to read CSV: {e}")
            return False
    
    def collect_node_ids(self, schema: NodeSchema) -> bool:
        """Collect all node IDs and check for duplicates."""
        if not self.validate_csv_structure(schema):
            return False
            
        filepath = self.csv_dir / schema.filename
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                line_num = 2  # Start after header
                
                seen_ids = set()
                row_count = 0
                
                for row in reader:
                    row_count += 1
                    node_id = row.get('id:ID', '').strip()
                    
                    if not node_id:
                        self.add_error(schema.filename, f"Empty id:ID", line_num)
                        continue
                    
                    # Check for duplicates within file
                    if node_id in seen_ids:
                        self.add_error(schema.filename, f"Duplicate ID: {node_id}", line_num)
                        continue
                        
                    seen_ids.add(node_id)
                    
                    # Check for duplicates across all files
                    if node_id in self.all_node_ids:
                        self.add_error(schema.filename, f"Duplicate ID across files: {node_id}", line_num)
                    else:
                        self.all_node_ids.add(node_id)
                        self.node_ids[schema.label].add(node_id)
                    
                    line_num += 1
                
                if row_count == 0:
                    self.add_warning(schema.filename, "File contains only headers (no data rows)")
                
                return True
                
        except Exception as e:
            self.add_error(schema.filename, f"Failed to process node IDs: {e}")
            return False
    
    def validate_relationship_file(self, schema: RelationshipSchema) -> bool:
        """Validate relationship file and check references."""
        if not self.validate_file_exists(schema.filename):
            return False
            
        filepath = self.csv_dir / schema.filename
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check headers
                if ':START_ID' not in reader.fieldnames:
                    self.add_error(schema.filename, "Missing :START_ID header")
                    return False
                if ':END_ID' not in reader.fieldnames:
                    self.add_error(schema.filename, "Missing :END_ID header")
                    return False
                
                line_num = 2  # Start after header
                missing_refs = []
                invalid_prefixes = []
                
                for row in reader:
                    start_id = row.get(':START_ID', '').strip()
                    end_id = row.get(':END_ID', '').strip()
                    
                    # Validate START_ID
                    if start_id:
                        if start_id not in self.all_node_ids:
                            missing_refs.append((line_num, 'START_ID', start_id))
                        elif schema.start_node_prefixes:
                            # Check prefix
                            prefix_valid = any(start_id.startswith(f"{prefix}:") for prefix in schema.start_node_prefixes if prefix)
                            if not prefix_valid and schema.start_node_prefixes != [""]:  # Allow no prefix if specified
                                invalid_prefixes.append((line_num, 'START_ID', start_id, schema.start_node_prefixes))
                    
                    # Validate END_ID  
                    if end_id:
                        if end_id not in self.all_node_ids:
                            missing_refs.append((line_num, 'END_ID', end_id))
                        elif schema.end_node_prefixes:
                            # Check prefix
                            prefix_valid = any(end_id.startswith(f"{prefix}:") for prefix in schema.end_node_prefixes if prefix)
                            if not prefix_valid and schema.end_node_prefixes != [""]:  # Allow no prefix if specified
                                invalid_prefixes.append((line_num, 'END_ID', end_id, schema.end_node_prefixes))
                    
                    line_num += 1
                
                # Report missing references (limit to first 10)
                for line_num, ref_type, ref_id in missing_refs[:10]:
                    self.add_error(schema.filename, f"Missing node reference: {ref_type} '{ref_id}'", line_num)
                
                if len(missing_refs) > 10:
                    self.add_error(schema.filename, f"... and {len(missing_refs) - 10} more missing references")
                
                # Report invalid prefixes (limit to first 5)
                for line_num, ref_type, ref_id, expected in invalid_prefixes[:5]:
                    self.add_warning(schema.filename, f"Unexpected prefix: {ref_type} '{ref_id}' (expected: {expected})", line_num)
                
                return len(missing_refs) == 0
                
        except Exception as e:
            self.add_error(schema.filename, f"Failed to validate relationships: {e}")
            return False
    
    def validate_data_integrity(self) -> bool:
        """Validate data integrity and consistency."""
        success = True
        
        # Check that we have reasonable data distributions
        for label, ids in self.node_ids.items():
            if len(ids) == 0:
                self.add_warning("schema", f"No {label} nodes found")
            elif len(ids) > 1000000:  # > 1M nodes might indicate an issue
                self.add_warning("schema", f"Very large node count for {label}: {len(ids):,}")
        
        return success
    
    def print_summary(self):
        """Print validation summary."""
        print(f"\n📊 Validation Summary for {self.csv_dir}")
        print("=" * 60)
        
        # Node counts
        print("\n📈 Node Counts:")
        total_nodes = 0
        for label, ids in self.node_ids.items():
            count = len(ids)
            total_nodes += count
            print(f"  {label:20} {count:>10,}")
        print(f"  {'Total':20} {total_nodes:>10,}")
        
        # Error summary
        print(f"\n🔍 Validation Results:")
        if not self.errors and not self.warnings:
            print("  ✅ All validations passed!")
        else:
            if self.errors:
                print(f"  ❌ Errors: {len(self.errors)}")
                for error in self.errors[:10]:  # Show first 10
                    location = f"{error.file}:{error.line}" if error.line else error.file
                    print(f"    {location} - {error.error}")
                if len(self.errors) > 10:
                    print(f"    ... and {len(self.errors) - 10} more errors")
            
            if self.warnings:
                print(f"  ⚠️  Warnings: {len(self.warnings)}")
                for warning in self.warnings[:5]:  # Show first 5
                    location = f"{warning.file}:{warning.line}" if warning.line else warning.file
                    print(f"    {location} - {warning.error}")
                if len(self.warnings) > 5:
                    print(f"    ... and {len(self.warnings) - 5} more warnings")
    
    def validate_all(self) -> bool:
        """Run complete validation suite."""
        print(f"🔍 Validating CSV schema in {self.csv_dir}")
        
        # Phase 1: Collect all node IDs and validate structure
        print("📋 Phase 1: Validating node files and collecting IDs...")
        for schema in self.node_schemas:
            self.collect_node_ids(schema)
        
        print(f"   Found {len(self.all_node_ids):,} total node IDs")
        
        # Phase 2: Validate relationships
        print("🔗 Phase 2: Validating relationship files...")
        for schema in self.relationship_schemas:
            self.validate_relationship_file(schema)
        
        # Phase 3: Data integrity checks
        print("🔎 Phase 3: Data integrity checks...")
        self.validate_data_integrity()
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0

def main():
    """Command line interface."""
    csv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/stage07_kg/csv")
    
    if not csv_dir.exists():
        print(f"❌ CSV directory not found: {csv_dir}")
        sys.exit(1)
    
    validator = CSVSchemaValidator(csv_dir)
    success = validator.validate_all()
    
    if success:
        print(f"\n✅ Schema validation passed! Ready for Neo4j import.")
        sys.exit(0)
    else:
        print(f"\n❌ Schema validation failed. Fix errors before attempting Neo4j import.")
        sys.exit(1)

if __name__ == "__main__":
    main()