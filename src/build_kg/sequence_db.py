#!/usr/bin/env python3
"""
Sequence Database Management

SQLite database for storing and retrieving protein sequences efficiently.
Provides fast sequence lookup without bloating the Neo4j knowledge graph.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SequenceDatabase:
    """SQLite database for protein sequence storage and retrieval."""
    
    def __init__(self, db_path: Path):
        """Initialize sequence database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sequences (
                    protein_id TEXT PRIMARY KEY,
                    sequence TEXT NOT NULL,
                    length INTEGER NOT NULL,
                    genome_id TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_genome_id ON sequences(genome_id);
                CREATE INDEX IF NOT EXISTS idx_length ON sequences(length);
                CREATE INDEX IF NOT EXISTS idx_source_file ON sequences(source_file);
                
                -- Metadata table for database info
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Insert database version
                INSERT OR REPLACE INTO metadata (key, value) 
                VALUES ('version', '1.0'), ('created_by', 'sequence_db.py');
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def insert_sequence(self, protein_id: str, sequence: str, genome_id: str, source_file: str) -> bool:
        """Insert a single sequence into the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sequences 
                    (protein_id, sequence, length, genome_id, source_file)
                    VALUES (?, ?, ?, ?, ?)
                """, (protein_id, sequence, len(sequence), genome_id, source_file))
            return True
        except Exception as e:
            logger.error(f"Failed to insert sequence {protein_id}: {e}")
            return False
    
    def insert_sequences_batch(self, sequences: List[Tuple[str, str, str, str]]) -> int:
        """Insert multiple sequences efficiently."""
        inserted = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for protein_id, sequence, genome_id, source_file in sequences:
                    cursor.execute("""
                        INSERT OR REPLACE INTO sequences 
                        (protein_id, sequence, length, genome_id, source_file)
                        VALUES (?, ?, ?, ?, ?)
                    """, (protein_id, sequence, len(sequence), genome_id, source_file))
                    inserted += cursor.rowcount
            logger.info(f"Inserted {inserted} sequences in batch")
            return inserted
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            return 0
    
    def get_sequence(self, protein_id: str) -> Optional[str]:
        """Get sequence for a single protein ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT sequence FROM sequences WHERE protein_id = ?", 
                    (protein_id,)
                )
                row = cursor.fetchone()
                return row['sequence'] if row else None
        except Exception as e:
            logger.error(f"Failed to get sequence for {protein_id}: {e}")
            return None
    
    def get_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """Get sequences for multiple protein IDs efficiently."""
        if not protein_ids:
            return {}
        
        try:
            with self._get_connection() as conn:
                # Use IN clause for efficient batch lookup
                placeholders = ','.join('?' * len(protein_ids))
                cursor = conn.execute(
                    f"SELECT protein_id, sequence FROM sequences WHERE protein_id IN ({placeholders})",
                    protein_ids
                )
                return {row['protein_id']: row['sequence'] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get sequences: {e}")
            return {}
    
    def get_sequences_by_genome(self, genome_id: str) -> Dict[str, str]:
        """Get all sequences for a specific genome."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT protein_id, sequence FROM sequences WHERE genome_id = ?",
                    (genome_id,)
                )
                return {row['protein_id']: row['sequence'] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get sequences for genome {genome_id}: {e}")
            return {}
    
    def search_sequences_by_pattern(self, pattern: str, limit: int = 100) -> List[Tuple[str, str]]:
        """Search sequences using SQL LIKE pattern matching."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT protein_id, sequence FROM sequences WHERE sequence LIKE ? LIMIT ?",
                    (f"%{pattern}%", limit)
                )
                return [(row['protein_id'], row['sequence']) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to search sequences: {e}")
            return []
    
    def get_protein_info(self, protein_id: str) -> Optional[Dict[str, Any]]:
        """Get complete information for a protein."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM sequences WHERE protein_id = ?",
                    (protein_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get protein info for {protein_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Total sequences
                cursor = conn.execute("SELECT COUNT(*) as count FROM sequences")
                stats['total_sequences'] = cursor.fetchone()['count']
                
                # Sequences by genome
                cursor = conn.execute("""
                    SELECT genome_id, COUNT(*) as count 
                    FROM sequences 
                    GROUP BY genome_id 
                    ORDER BY count DESC
                """)
                stats['sequences_by_genome'] = dict(cursor.fetchall())
                
                # Length statistics
                cursor = conn.execute("""
                    SELECT 
                        MIN(length) as min_length,
                        MAX(length) as max_length,
                        AVG(length) as avg_length,
                        COUNT(DISTINCT genome_id) as unique_genomes
                    FROM sequences
                """)
                length_stats = cursor.fetchone()
                stats.update(dict(length_stats))
                
                # Database size
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def protein_exists(self, protein_id: str) -> bool:
        """Check if a protein exists in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM sequences WHERE protein_id = ? LIMIT 1",
                    (protein_id,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check protein existence: {e}")
            return False
    
    def delete_sequences_by_genome(self, genome_id: str) -> int:
        """Delete all sequences for a genome (for re-processing)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM sequences WHERE genome_id = ?",
                    (genome_id,)
                )
                deleted = cursor.rowcount
                logger.info(f"Deleted {deleted} sequences for genome {genome_id}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete sequences for genome {genome_id}: {e}")
            return 0
    
    def vacuum(self):
        """Optimize database (reclaim space after deletions)."""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")


def get_default_sequence_db() -> SequenceDatabase:
    """Get default sequence database instance."""
    db_path = Path("data/sequences.db")
    return SequenceDatabase(db_path)