#!/usr/bin/env python3
"""
Sequence Database Builder

Builds SQLite sequence database from prodigal FASTA files.
Efficiently processes all protein sequences from the genomic pipeline.
"""

import re
import logging
from pathlib import Path
from typing import Generator, Tuple, List, Dict, Set
from Bio import SeqIO
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .sequence_db import SequenceDatabase

console = Console()
logger = logging.getLogger(__name__)

class SequenceDatabaseBuilder:
    """Builder for protein sequence SQLite database."""
    
    def __init__(self, db_path: Path):
        """Initialize builder with database path."""
        self.db = SequenceDatabase(db_path)
        self.stats = {
            'files_processed': 0,
            'sequences_added': 0,
            'sequences_updated': 0,
            'errors': 0
        }
    
    def parse_prodigal_header(self, header: str) -> Tuple[str, Dict[str, str]]:
        """
        Parse prodigal FASTA header to extract protein ID and metadata.
        
        Example header:
        >protein_id # 76 # 171 # -1 # ID=1_1;partial=00;start_type=ATG;rbs_motif=AGGAG;rbs_spacer=5-10bp;gc_cont=0.573
        
        Returns:
            tuple: (protein_id, metadata_dict)
        """
        try:
            parts = header.split('#')
            if len(parts) < 4:
                logger.warning(f"Unexpected header format: {header}")
                return header.strip('>'), {}
            
            protein_id = parts[0].strip().lstrip('>')
            start_pos = parts[1].strip()
            end_pos = parts[2].strip()
            strand = parts[3].strip()
            
            metadata = {
                'start_pos': int(start_pos) if start_pos.isdigit() else None,
                'end_pos': int(end_pos) if end_pos.isdigit() else None,
                'strand': int(strand) if strand.lstrip('-').isdigit() else None
            }
            
            # Parse additional metadata if present
            if len(parts) > 4:
                extra_info = parts[4].strip()
                for item in extra_info.split(';'):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        metadata[key.strip()] = value.strip()
            
            return protein_id, metadata
            
        except Exception as e:
            logger.error(f"Failed to parse header '{header}': {e}")
            return header.strip('>'), {}
    
    def extract_genome_id_from_path(self, fasta_path: Path) -> str:
        """Extract genome ID from FASTA file path."""
        # Expected pattern: data/stage03_prodigal/genome_name.faa
        stem = fasta_path.stem
        # Remove common suffixes
        for suffix in ['.proteins', '.faa', '_proteins']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        return stem
    
    def process_fasta_file(self, fasta_path: Path) -> Generator[Tuple[str, str, str, str], None, None]:
        """
        Process a single FASTA file and yield sequence data.
        
        Yields:
            tuple: (protein_id, sequence, genome_id, source_file)
        """
        genome_id = self.extract_genome_id_from_path(fasta_path)
        source_file = str(fasta_path.relative_to(Path.cwd()))
        
        try:
            with open(fasta_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    protein_id, metadata = self.parse_prodigal_header(record.description)
                    sequence = str(record.seq)
                    
                    # Validate sequence (basic checks)
                    if not sequence or len(sequence) < 10:
                        logger.warning(f"Skipping short/empty sequence: {protein_id}")
                        continue
                    
                    # Check for valid amino acid characters
                    valid_aa = set('ACDEFGHIKLMNPQRSTVWY*X')
                    if not set(sequence.upper()).issubset(valid_aa):
                        logger.warning(f"Invalid amino acids in sequence: {protein_id}")
                        continue
                    
                    yield protein_id, sequence, genome_id, source_file
                    
        except Exception as e:
            logger.error(f"Failed to process {fasta_path}: {e}")
            self.stats['errors'] += 1
    
    def build_from_prodigal_directory(self, prodigal_dir: Path, batch_size: int = 1000) -> bool:
        """
        Build sequence database from prodigal output directory.
        
        Args:
            prodigal_dir: Directory containing .faa files
            batch_size: Number of sequences to insert per batch
            
        Returns:
            bool: Success status
        """
        prodigal_dir = Path(prodigal_dir)
        if not prodigal_dir.exists():
            logger.error(f"Prodigal directory not found: {prodigal_dir}")
            return False
        
        # Find all FASTA files
        fasta_files = list(prodigal_dir.glob("*.faa"))
        if not fasta_files:
            logger.warning(f"No .faa files found in {prodigal_dir}")
            return False
        
        console.print(f"🧬 Building sequence database from {len(fasta_files)} FASTA files")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            main_task = progress.add_task("Processing files...", total=len(fasta_files))
            
            batch = []
            total_sequences = 0
            
            for fasta_path in fasta_files:
                progress.update(main_task, description=f"Processing {fasta_path.name}")
                
                try:
                    file_sequences = 0
                    for sequence_data in self.process_fasta_file(fasta_path):
                        batch.append(sequence_data)
                        file_sequences += 1
                        
                        # Insert batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            inserted = self.db.insert_sequences_batch(batch)
                            self.stats['sequences_added'] += inserted
                            total_sequences += inserted
                            batch = []
                    
                    self.stats['files_processed'] += 1
                    logger.info(f"Processed {fasta_path.name}: {file_sequences} sequences")
                    
                except Exception as e:
                    logger.error(f"Failed to process {fasta_path}: {e}")
                    self.stats['errors'] += 1
                
                progress.advance(main_task)
            
            # Insert remaining sequences in batch
            if batch:
                inserted = self.db.insert_sequences_batch(batch)
                self.stats['sequences_added'] += inserted
                total_sequences += inserted
        
        # Get final statistics
        db_stats = self.db.get_statistics()
        
        console.print(f"\n✅ Database build completed!")
        console.print(f"   Files processed: {self.stats['files_processed']}")
        console.print(f"   Sequences added: {self.stats['sequences_added']}")
        console.print(f"   Total in database: {db_stats.get('total_sequences', 0)}")
        console.print(f"   Database size: {db_stats.get('database_size_mb', 0):.1f} MB")
        console.print(f"   Unique genomes: {db_stats.get('unique_genomes', 0)}")
        
        if self.stats['errors'] > 0:
            console.print(f"   ⚠️  Errors encountered: {self.stats['errors']}")
        
        return True
    
    def update_from_new_files(self, new_fasta_files: List[Path]) -> bool:
        """Update database with new FASTA files (incremental)."""
        if not new_fasta_files:
            return True
        
        console.print(f"🔄 Updating database with {len(new_fasta_files)} new files")
        
        for fasta_path in new_fasta_files:
            # Check if we already have sequences from this file
            genome_id = self.extract_genome_id_from_path(fasta_path)
            existing_sequences = self.db.get_sequences_by_genome(genome_id)
            
            if existing_sequences:
                console.print(f"   Updating genome {genome_id} ({len(existing_sequences)} existing sequences)")
                # Delete existing sequences for this genome
                self.db.delete_sequences_by_genome(genome_id)
            
            # Process new sequences
            batch = []
            for sequence_data in self.process_fasta_file(fasta_path):
                batch.append(sequence_data)
            
            if batch:
                inserted = self.db.insert_sequences_batch(batch)
                self.stats['sequences_added'] += inserted
                console.print(f"   Added {inserted} sequences from {fasta_path.name}")
        
        return True
    
    def get_build_statistics(self) -> Dict[str, int]:
        """Get build statistics."""
        return self.stats.copy()


def main():
    """CLI entry point for building sequence database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build protein sequence database from prodigal output")
    parser.add_argument("--input", "-i", type=Path, default="data/stage03_prodigal",
                        help="Input directory with .faa files")
    parser.add_argument("--output", "-o", type=Path, default="data/sequences.db",
                        help="Output SQLite database path")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for database inserts")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if database exists")
    parser.add_argument("--stats", action="store_true",
                        help="Show database statistics only")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.stats:
        # Show statistics only
        if args.output.exists():
            db = SequenceDatabase(args.output)
            stats = db.get_statistics()
            console.print("📊 Database Statistics:")
            for key, value in stats.items():
                console.print(f"   {key}: {value}")
        else:
            console.print("❌ Database does not exist")
        return
    
    # Check if database exists
    if args.output.exists() and not args.force:
        console.print(f"❌ Database already exists: {args.output}")
        console.print("   Use --force to rebuild or --stats to view current statistics")
        return
    
    # Build database
    builder = SequenceDatabaseBuilder(args.output)
    success = builder.build_from_prodigal_directory(args.input, args.batch_size)
    
    if success:
        console.print(f"🎉 Sequence database ready: {args.output}")
    else:
        console.print("❌ Failed to build sequence database")
        exit(1)


if __name__ == "__main__":
    main()