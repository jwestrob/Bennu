#!/usr/bin/env python3
"""
Session Results Accumulator for Agentic Discovery

This module provides persistent storage for curated biological findings
discovered during agentic workflows, separate from the task execution context.
Agents push discoveries here, and synthesis pulls the structured results.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class DiscoveryType(Enum):
    PROPHAGE_CANDIDATE = "prophage_candidate"
    OPERON_PREDICTION = "operon_prediction"
    HYPOTHETICAL_STRETCH = "hypothetical_stretch"
    NOVEL_GENE_CLUSTER = "novel_gene_cluster"
    BGC_CANDIDATE = "bgc_candidate"
    SPATIAL_PATTERN = "spatial_pattern"

@dataclass
class ProphageCandidate:
    """Potential prophage region discovered through spatial analysis."""
    genome_id: str
    contig: str
    start_coordinate: int
    end_coordinate: int
    gene_count: int
    hypothetical_count: int
    hypothetical_percentage: float
    potential_phage_proteins: List[str]
    gc_deviation: Optional[float] = None
    integrase_present: bool = False
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    discovered_by_task: str = ""
    discovery_timestamp: str = ""
    
    def __post_init__(self):
        if not self.discovery_timestamp:
            self.discovery_timestamp = datetime.now().isoformat()

@dataclass
class HypotheticalStretch:
    """Stretch of consecutive hypothetical/unannotated genes."""
    genome_id: str
    contig: str
    start_coordinate: int
    end_coordinate: int
    gene_count: int
    gene_ids: List[str]
    avg_gene_length: float
    strand_consistency: bool = False
    potential_operon: bool = False
    discovered_by_task: str = ""
    discovery_timestamp: str = ""
    
    def __post_init__(self):
        if not self.discovery_timestamp:
            self.discovery_timestamp = datetime.now().isoformat()

@dataclass
class OperonPrediction:
    """Predicted operon with functional coherence."""
    genome_id: str
    contig: str
    start_coordinate: int
    end_coordinate: int
    gene_count: int
    gene_ids: List[str]
    functional_category: str
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    intergenic_distances: List[int] = None
    strand_coherence: bool = True
    discovered_by_task: str = ""
    discovery_timestamp: str = ""
    
    def __post_init__(self):
        if not self.discovery_timestamp:
            self.discovery_timestamp = datetime.now().isoformat()

@dataclass
class SpatialPattern:
    """General spatial genomic pattern of interest."""
    genome_id: str
    contig: str
    pattern_type: str
    description: str
    coordinates: List[tuple]  # [(start, end), ...]
    evidence: Dict[str, Any]
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    discovered_by_task: str = ""
    discovery_timestamp: str = ""
    
    def __post_init__(self):
        if not self.discovery_timestamp:
            self.discovery_timestamp = datetime.now().isoformat()

class SessionResultsAccumulator:
    """
    Persistent accumulator for biological discoveries during agentic analysis.
    
    Agents write discoveries here during execution, and synthesis reads
    the curated results for final answer generation.
    """
    
    def __init__(self, session_id: str, notes_folder: Path):
        """
        Initialize accumulator for a session.
        
        Args:
            session_id: Unique session identifier
            notes_folder: Session notes folder path
        """
        self.session_id = session_id
        self.notes_folder = Path(notes_folder)
        self.results_file = self.notes_folder / "discovery_results.json"
        
        # Initialize discovery storage
        self.prophage_candidates: List[ProphageCandidate] = []
        self.hypothetical_stretches: List[HypotheticalStretch] = []
        self.operon_predictions: List[OperonPrediction] = []
        self.spatial_patterns: List[SpatialPattern] = []
        
        # Load existing results if file exists
        self._load_existing_results()
        
        logger.info(f"SessionResultsAccumulator initialized for session {session_id}")
    
    def add_prophage_candidate(self, 
                             genome_id: str,
                             contig: str, 
                             start: int,
                             end: int,
                             gene_count: int,
                             hypothetical_count: int,
                             evidence: Dict[str, Any],
                             task_id: str = "",
                             confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM) -> None:
        """Add a prophage candidate discovery."""
        
        hypothetical_percentage = (hypothetical_count / gene_count * 100) if gene_count > 0 else 0
        
        candidate = ProphageCandidate(
            genome_id=genome_id,
            contig=contig,
            start_coordinate=start,
            end_coordinate=end,
            gene_count=gene_count,
            hypothetical_count=hypothetical_count,
            hypothetical_percentage=hypothetical_percentage,
            potential_phage_proteins=evidence.get("phage_proteins", []),
            gc_deviation=evidence.get("gc_deviation"),
            integrase_present=evidence.get("integrase_present", False),
            confidence=confidence,
            discovered_by_task=task_id
        )
        
        self.prophage_candidates.append(candidate)
        self._save_results()
        
        logger.info(f"🧬 Added prophage candidate: {genome_id}:{contig}:{start}-{end} ({hypothetical_count}/{gene_count} hypothetical)")
    
    def add_hypothetical_stretch(self,
                               genome_id: str,
                               contig: str,
                               start: int, 
                               end: int,
                               gene_ids: List[str],
                               task_id: str = "") -> None:
        """Add a hypothetical gene stretch discovery."""
        
        gene_count = len(gene_ids)
        avg_length = (end - start) / gene_count if gene_count > 0 else 0
        
        stretch = HypotheticalStretch(
            genome_id=genome_id,
            contig=contig,
            start_coordinate=start,
            end_coordinate=end,
            gene_count=gene_count,
            gene_ids=gene_ids,
            avg_gene_length=avg_length,
            discovered_by_task=task_id
        )
        
        self.hypothetical_stretches.append(stretch)
        self._save_results()
        
        logger.info(f"🔍 Added hypothetical stretch: {genome_id}:{contig}:{start}-{end} ({gene_count} genes)")
    
    def add_operon_prediction(self,
                            genome_id: str,
                            contig: str,
                            start: int,
                            end: int, 
                            gene_ids: List[str],
                            functional_category: str,
                            confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                            task_id: str = "") -> None:
        """Add an operon prediction."""
        
        operon = OperonPrediction(
            genome_id=genome_id,
            contig=contig,
            start_coordinate=start,
            end_coordinate=end,
            gene_count=len(gene_ids),
            gene_ids=gene_ids,
            functional_category=functional_category,
            confidence=confidence,
            discovered_by_task=task_id
        )
        
        self.operon_predictions.append(operon)
        self._save_results()
        
        logger.info(f"🔗 Added operon prediction: {genome_id}:{contig}:{start}-{end} ({functional_category})")
    
    def add_spatial_pattern(self,
                          genome_id: str,
                          contig: str,
                          pattern_type: str,
                          description: str,
                          coordinates: List[tuple],
                          evidence: Dict[str, Any],
                          confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                          task_id: str = "") -> None:
        """Add a general spatial pattern discovery."""
        
        pattern = SpatialPattern(
            genome_id=genome_id,
            contig=contig,
            pattern_type=pattern_type,
            description=description,
            coordinates=coordinates,
            evidence=evidence,
            confidence=confidence,
            discovered_by_task=task_id
        )
        
        self.spatial_patterns.append(pattern)
        self._save_results()
        
        logger.info(f"📊 Added spatial pattern: {genome_id}:{contig} ({pattern_type})")
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get structured summary of all discoveries for synthesis."""
        
        summary = {
            "session_id": self.session_id,
            "total_discoveries": (
                len(self.prophage_candidates) + 
                len(self.hypothetical_stretches) + 
                len(self.operon_predictions) + 
                len(self.spatial_patterns)
            ),
            "prophage_candidates": {
                "count": len(self.prophage_candidates),
                "high_confidence": len([c for c in self.prophage_candidates if c.confidence == ConfidenceLevel.HIGH]),
                "candidates": [asdict(c) for c in self.prophage_candidates]
            },
            "hypothetical_stretches": {
                "count": len(self.hypothetical_stretches),
                "stretches": [asdict(s) for s in self.hypothetical_stretches]
            },
            "operon_predictions": {
                "count": len(self.operon_predictions),
                "predictions": [asdict(o) for o in self.operon_predictions]
            },
            "spatial_patterns": {
                "count": len(self.spatial_patterns),
                "patterns": [asdict(p) for p in self.spatial_patterns]
            },
            "genomes_analyzed": len(set(
                [c.genome_id for c in self.prophage_candidates] +
                [s.genome_id for s in self.hypothetical_stretches] +
                [o.genome_id for o in self.operon_predictions] +
                [p.genome_id for p in self.spatial_patterns]
            ))
        }
        
        return summary
    
    def get_synthesis_context(self) -> str:
        """Generate structured context for final synthesis."""
        
        summary = self.get_discovery_summary()
        
        context_lines = [
            f"Agentic Genomic Discovery Results (Session: {self.session_id})",
            f"Total discoveries: {summary['total_discoveries']} across {summary['genomes_analyzed']} genomes",
            ""
        ]
        
        # Prophage candidates
        if summary['prophage_candidates']['count'] > 0:
            context_lines.extend([
                f"PROPHAGE CANDIDATES: {summary['prophage_candidates']['count']} found",
                f"High confidence: {summary['prophage_candidates']['high_confidence']}"
            ])
            
            for candidate in summary['prophage_candidates']['candidates']:
                context_lines.append(
                    f"- {candidate['genome_id']}:{candidate['contig']}:"
                    f"{candidate['start_coordinate']}-{candidate['end_coordinate']} "
                    f"({candidate['hypothetical_count']}/{candidate['gene_count']} hypothetical, "
                    f"{candidate['hypothetical_percentage']:.1f}%, "
                    f"confidence: {candidate['confidence']})"
                )
            
            context_lines.append("")
        
        # Hypothetical stretches
        if summary['hypothetical_stretches']['count'] > 0:
            context_lines.extend([
                f"HYPOTHETICAL GENE STRETCHES: {summary['hypothetical_stretches']['count']} found"
            ])
            
            for stretch in summary['hypothetical_stretches']['stretches']:
                context_lines.append(
                    f"- {stretch['genome_id']}:{stretch['contig']}:"
                    f"{stretch['start_coordinate']}-{stretch['end_coordinate']} "
                    f"({stretch['gene_count']} genes)"
                )
            
            context_lines.append("")
        
        # Operon predictions  
        if summary['operon_predictions']['count'] > 0:
            context_lines.extend([
                f"OPERON PREDICTIONS: {summary['operon_predictions']['count']} found"
            ])
            
            for operon in summary['operon_predictions']['predictions']:
                context_lines.append(
                    f"- {operon['genome_id']}:{operon['contig']}:"
                    f"{operon['start_coordinate']}-{operon['end_coordinate']} "
                    f"({operon['functional_category']}, {operon['confidence']})"
                )
                
            context_lines.append("")
        
        # Add footer
        context_lines.extend([
            f"Detailed evidence and gene lists available in session notes: {self.session_id}",
            "Analysis complete."
        ])
        
        return "\n".join(context_lines)
    
    def _load_existing_results(self) -> None:
        """Load existing results from file if it exists."""
        
        if not self.results_file.exists():
            return
            
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            # Load prophage candidates
            for item in data.get('prophage_candidates', []):
                candidate = ProphageCandidate(**item)
                self.prophage_candidates.append(candidate)
            
            # Load hypothetical stretches
            for item in data.get('hypothetical_stretches', []):
                stretch = HypotheticalStretch(**item)
                self.hypothetical_stretches.append(stretch)
                
            # Load operon predictions
            for item in data.get('operon_predictions', []):
                operon = OperonPrediction(**item)
                self.operon_predictions.append(operon)
                
            # Load spatial patterns
            for item in data.get('spatial_patterns', []):
                pattern = SpatialPattern(**item)
                self.spatial_patterns.append(pattern)
            
            logger.info(f"Loaded existing results: {len(self.prophage_candidates)} prophage candidates, "
                       f"{len(self.hypothetical_stretches)} hypothetical stretches, "
                       f"{len(self.operon_predictions)} operons, "
                       f"{len(self.spatial_patterns)} patterns")
                       
        except Exception as e:
            logger.error(f"Failed to load existing results: {e}")
    
    def _save_results(self) -> None:
        """Save current results to file."""
        
        try:
            data = {
                'session_id': self.session_id,
                'last_updated': datetime.now().isoformat(),
                'prophage_candidates': [asdict(c) for c in self.prophage_candidates],
                'hypothetical_stretches': [asdict(s) for s in self.hypothetical_stretches], 
                'operon_predictions': [asdict(o) for o in self.operon_predictions],
                'spatial_patterns': [asdict(p) for p in self.spatial_patterns]
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")