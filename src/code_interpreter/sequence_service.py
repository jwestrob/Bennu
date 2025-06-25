#!/usr/bin/env python3
"""
Sequence Service for Code Interpreter

Provides protein sequence access for the code interpreter service.
Integrates with SQLite sequence database for fast lookups.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import sequence database directly for Docker environment
import sys
from pathlib import Path
sys.path.append('/app/build_kg')

try:
    from sequence_db import SequenceDatabase
except ImportError:
    # Fallback for development environment
    from ..build_kg.sequence_db import SequenceDatabase

def get_default_sequence_db():
    """Get default sequence database instance."""
    db_path = Path("/app/sequences.db")
    if db_path.exists():
        return SequenceDatabase(db_path, read_only=True)  # Read-only for Docker mount
    else:
        raise FileNotFoundError(f"Sequence database not found at {db_path}")

logger = logging.getLogger(__name__)

@dataclass
class SequenceInfo:
    """Protein sequence information."""
    protein_id: str
    sequence: str
    length: int
    genome_id: str
    source_file: str

class SequenceService:
    """Async service for protein sequence retrieval."""
    
    def __init__(self, db_path: Optional[Path] = None, read_only: bool = False):
        """Initialize sequence service."""
        if db_path:
            self.db = SequenceDatabase(db_path, read_only=read_only)
        else:
            self.db = get_default_sequence_db()
        self._cache = {}  # Simple in-memory cache
        self._cache_size_limit = 1000
    
    async def get_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """
        Get sequences for multiple protein IDs.
        
        Args:
            protein_ids: List of protein identifiers
            
        Returns:
            Dict mapping protein_id to sequence
        """
        if not protein_ids:
            return {}
        
        # Check cache first
        cached_results = {}
        remaining_ids = []
        
        for protein_id in protein_ids:
            if protein_id in self._cache:
                cached_results[protein_id] = self._cache[protein_id]
            else:
                remaining_ids.append(protein_id)
        
        # Fetch remaining from database
        if remaining_ids:
            # Run database query in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            db_results = await loop.run_in_executor(
                None, self.db.get_sequences, remaining_ids
            )
            
            # Update cache
            for protein_id, sequence in db_results.items():
                self._update_cache(protein_id, sequence)
            
            # Combine results
            cached_results.update(db_results)
        
        logger.info(f"Retrieved {len(cached_results)} sequences for {len(protein_ids)} requested IDs")
        return cached_results
    
    async def get_sequence(self, protein_id: str) -> Optional[str]:
        """Get sequence for a single protein ID."""
        results = await self.get_sequences([protein_id])
        return results.get(protein_id)
    
    async def get_sequences_by_genome(self, genome_id: str) -> Dict[str, str]:
        """Get all sequences for a specific genome."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.db.get_sequences_by_genome, genome_id
        )
        
        # Update cache with genome sequences
        for protein_id, sequence in result.items():
            self._update_cache(protein_id, sequence)
        
        logger.info(f"Retrieved {len(result)} sequences for genome {genome_id}")
        return result
    
    async def search_by_pattern(self, pattern: str, limit: int = 100) -> List[Tuple[str, str]]:
        """
        Search sequences by pattern.
        
        Args:
            pattern: SQL LIKE pattern to search for
            limit: Maximum number of results
            
        Returns:
            List of (protein_id, sequence) tuples
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self.db.search_sequences_by_pattern, pattern, limit
        )
        
        logger.info(f"Pattern search '{pattern}' returned {len(results)} results")
        return results
    
    async def get_protein_info(self, protein_id: str) -> Optional[SequenceInfo]:
        """Get complete protein information."""
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            None, self.db.get_protein_info, protein_id
        )
        
        if info:
            return SequenceInfo(
                protein_id=info['protein_id'],
                sequence=info['sequence'],
                length=info['length'],
                genome_id=info['genome_id'],
                source_file=info['source_file']
            )
        return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self.db.get_statistics)
        return stats
    
    async def protein_exists(self, protein_id: str) -> bool:
        """Check if protein exists in database."""
        # Check cache first
        if protein_id in self._cache:
            return True
        
        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(
            None, self.db.protein_exists, protein_id
        )
        return exists
    
    async def batch_exists_check(self, protein_ids: List[str]) -> Dict[str, bool]:
        """Check existence of multiple proteins efficiently."""
        # This could be optimized with a dedicated database method
        results = {}
        
        # Check cache first
        remaining_ids = []
        for protein_id in protein_ids:
            if protein_id in self._cache:
                results[protein_id] = True
            else:
                remaining_ids.append(protein_id)
        
        # Check remaining in database
        if remaining_ids:
            sequences = await self.get_sequences(remaining_ids)
            for protein_id in remaining_ids:
                results[protein_id] = protein_id in sequences
        
        return results
    
    def _update_cache(self, protein_id: str, sequence: str):
        """Update in-memory cache with size limit."""
        if len(self._cache) >= self._cache_size_limit:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[protein_id] = sequence
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("Sequence cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'cache_limit': self._cache_size_limit,
            'cache_usage_percent': (len(self._cache) / self._cache_size_limit) * 100
        }


class SequenceAnalyzer:
    """Helper class for sequence analysis operations."""
    
    def __init__(self, sequence_service: SequenceService):
        """Initialize with sequence service."""
        self.service = sequence_service
    
    async def calculate_amino_acid_composition(self, protein_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate amino acid composition for proteins.
        
        Returns:
            Dict mapping protein_id to amino acid frequencies
        """
        sequences = await self.service.get_sequences(protein_ids)
        compositions = {}
        
        for protein_id, sequence in sequences.items():
            composition = self._calculate_aa_composition(sequence)
            compositions[protein_id] = composition
        
        return compositions
    
    def _calculate_aa_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate amino acid composition for a single sequence."""
        aa_count = {}
        clean_sequence = sequence.upper().replace('*', '')  # Remove stop codons
        
        if not clean_sequence:
            return {}
        
        # Count amino acids
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            count = clean_sequence.count(aa)
            aa_count[aa] = count / len(clean_sequence)
        
        return aa_count
    
    async def calculate_hydrophobicity_profiles(self, protein_ids: List[str]) -> Dict[str, float]:
        """
        Calculate Kyte-Doolittle hydrophobicity scores.
        
        Returns:
            Dict mapping protein_id to average hydrophobicity
        """
        # Kyte-Doolittle hydrophobicity scale
        kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        sequences = await self.service.get_sequences(protein_ids)
        hydrophobicity = {}
        
        for protein_id, sequence in sequences.items():
            clean_sequence = sequence.upper().replace('*', '')
            if clean_sequence:
                scores = [kd_scale.get(aa, 0) for aa in clean_sequence]
                avg_hydrophobicity = sum(scores) / len(scores)
                hydrophobicity[protein_id] = avg_hydrophobicity
            else:
                hydrophobicity[protein_id] = 0.0
        
        return hydrophobicity
    
    async def group_sequences_by_length(self, protein_ids: List[str], bins: List[int] = None) -> Dict[str, List[str]]:
        """Group proteins by sequence length ranges."""
        if bins is None:
            bins = [0, 100, 200, 300, 500, 1000, float('inf')]
        
        sequences = await self.service.get_sequences(protein_ids)
        groups = {f"{bins[i]}-{bins[i+1]}": [] for i in range(len(bins)-1)}
        
        for protein_id, sequence in sequences.items():
            length = len(sequence)
            for i in range(len(bins)-1):
                if bins[i] <= length < bins[i+1]:
                    group_name = f"{bins[i]}-{bins[i+1]}"
                    groups[group_name].append(protein_id)
                    break
        
        return groups


# Global service instance for easy access
_sequence_service = None

def get_sequence_service(db_path: Optional[Path] = None) -> SequenceService:
    """Get global sequence service instance."""
    global _sequence_service
    if _sequence_service is None:
        _sequence_service = SequenceService(db_path)
    return _sequence_service


# Convenience functions for common operations
async def fetch_sequences(protein_ids: List[str]) -> Dict[str, str]:
    """Convenience function to fetch sequences."""
    service = get_sequence_service()
    return await service.get_sequences(protein_ids)

async def fetch_sequence(protein_id: str) -> Optional[str]:
    """Convenience function to fetch single sequence."""
    service = get_sequence_service()
    return await service.get_sequence(protein_id)

async def calculate_aa_composition(protein_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Convenience function for amino acid composition analysis."""
    service = get_sequence_service()
    analyzer = SequenceAnalyzer(service)
    return await analyzer.calculate_amino_acid_composition(protein_ids)

async def calculate_hydrophobicity(protein_ids: List[str]) -> Dict[str, float]:
    """Convenience function for hydrophobicity analysis."""
    service = get_sequence_service()
    analyzer = SequenceAnalyzer(service)
    return await analyzer.calculate_hydrophobicity_profiles(protein_ids)