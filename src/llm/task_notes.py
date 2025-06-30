#!/usr/bin/env python3
"""
Task Notes System for Large Dataset Processing
Enables multi-task workflows to handle large datasets through structured note-taking
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class TaskNotesManager:
    """Manages structured notes for large dataset processing across agentic tasks"""
    
    def __init__(self, notes_dir: str = "data/task_notes"):
        self.notes_dir = Path(notes_dir)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = None
        self.current_notes = {}
    
    def start_session(self, query: str) -> str:
        """Start a new note-taking session for a query"""
        # Create session ID from query hash + timestamp
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp}_{query_hash}"
        
        # Initialize session notes
        self.current_notes = {
            "session_id": self.session_id,
            "query": query,
            "started_at": datetime.now().isoformat(),
            "tasks": {},
            "summary": {},
            "statistics": {}
        }
        
        logger.info(f"📝 Started notes session: {self.session_id}")
        return self.session_id
    
    def add_task_notes(self, task_id: str, task_type: str, data: Dict[str, Any], 
                      summary: Optional[str] = None) -> None:
        """Add structured notes for a task result"""
        if not self.session_id:
            logger.warning("No active session - starting new session")
            self.start_session("Unknown query")
        
        # Create task notes with summary and key statistics
        task_notes = {
            "task_id": task_id,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "summary": summary or f"Completed {task_type} task",
            "data_summary": self._summarize_data(data),
            "key_findings": self._extract_key_findings(data, task_type),
            "statistics": self._calculate_statistics(data, task_type)
        }
        
        # Store only essential data, not full results
        if task_type == "pathway_discovery":
            task_notes["proteins_found"] = data.get("total_proteins", 0)
            task_notes["pathways_analyzed"] = data.get("pathways_analyzed", 0)
            task_notes["top_proteins"] = self._get_top_proteins(data)
            task_notes["pathway_summary"] = self._get_pathway_summary(data)
        
        elif task_type == "annotation_analysis":
            task_notes["annotations_processed"] = len(data.get("annotations", []))
            task_notes["functional_categories"] = self._get_functional_categories(data)
            task_notes["top_annotations"] = self._get_top_annotations(data)
        
        elif task_type == "sequence_analysis":
            task_notes["sequences_analyzed"] = len(data.get("sequences", []))
            task_notes["analysis_metrics"] = self._get_sequence_metrics(data)
        
        self.current_notes["tasks"][task_id] = task_notes
        logger.info(f"📝 Added notes for task {task_id}: {task_notes['summary']}")
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Create a concise summary of data structure"""
        if not data:
            return "Empty dataset"
        
        summary_parts = []
        
        # Count major data types
        for key, value in data.items():
            if isinstance(value, list):
                summary_parts.append(f"{len(value)} {key}")
            elif isinstance(value, dict):
                summary_parts.append(f"{len(value)} {key} entries")
            elif isinstance(value, (int, float)):
                summary_parts.append(f"{key}={value}")
        
        return ", ".join(summary_parts[:5])  # Limit to top 5 items
    
    def _extract_key_findings(self, data: Dict[str, Any], task_type: str) -> List[str]:
        """Extract key findings from task data"""
        findings = []
        
        if task_type == "pathway_discovery":
            if "proteins_found" in data:
                proteins = data["proteins_found"]
                if proteins:
                    # Group by KO function
                    ko_counts = {}
                    for protein in proteins:
                        ko_desc = protein.get("ko_description", "Unknown")
                        ko_counts[ko_desc] = ko_counts.get(ko_desc, 0) + 1
                    
                    # Top 3 functions
                    top_functions = sorted(ko_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func, count in top_functions:
                        findings.append(f"{count} proteins with {func[:50]}...")
        
        elif task_type == "annotation_analysis":
            if "classification" in data:
                classification = data["classification"]
                for category, annotations in classification.items():
                    if annotations:
                        findings.append(f"{len(annotations)} {category.lower()} annotations")
        
        return findings[:5]  # Limit to top 5 findings
    
    def _calculate_statistics(self, data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Calculate key statistics for the data"""
        stats = {}
        
        if task_type == "pathway_discovery":
            if "proteins_found" in data:
                proteins = data["proteins_found"]
                stats["total_proteins"] = len(proteins)
                
                # Pathway distribution
                pathway_counts = {}
                for protein in proteins:
                    pathway = protein.get("pathway_id", "Unknown")
                    pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
                stats["pathways_represented"] = len(pathway_counts)
                stats["avg_proteins_per_pathway"] = len(proteins) / max(1, len(pathway_counts))
        
        return stats
    
    def _get_top_proteins(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get top proteins with essential info"""
        proteins = data.get("proteins_found", [])
        return [
            {
                "protein_id": p.get("protein_id", ""),
                "ko_id": p.get("ko_id", ""),
                "function": p.get("ko_description", "")[:50] + "..." if len(p.get("ko_description", "")) > 50 else p.get("ko_description", "")
            }
            for p in proteins[:10]  # Top 10 proteins
        ]
    
    def _get_pathway_summary(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get pathway summary information"""
        pathway_details = data.get("pathway_details", [])
        return [
            {
                "pathway_id": p.get("pathway_id", ""),
                "proteins_found": p.get("proteins_found", 0),
                "relevance_score": p.get("relevance_score", 0)
            }
            for p in pathway_details
        ]
    
    def _get_functional_categories(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Get functional category distribution"""
        classification = data.get("classification", {})
        return {category: len(annotations) for category, annotations in classification.items()}
    
    def _get_top_annotations(self, data: Dict[str, Any]) -> List[str]:
        """Get top annotations"""
        relevant = data.get("classification", {}).get("RELEVANT", [])
        return relevant[:10]  # Top 10 relevant annotations
    
    def _get_sequence_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get sequence analysis metrics"""
        sequences = data.get("sequences", [])
        if not sequences:
            return {}
        
        return {
            "total_sequences": len(sequences),
            "avg_length": sum(len(seq.get("sequence", "")) for seq in sequences) / len(sequences),
            "organisms_represented": len(set(seq.get("organism", "") for seq in sequences))
        }
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive session summary for LLM processing"""
        if not self.current_notes:
            return {"error": "No active session"}
        
        # Calculate overall statistics
        total_proteins = 0
        total_pathways = 0
        total_annotations = 0
        
        key_findings = []
        task_summaries = []
        
        for task_id, task_notes in self.current_notes["tasks"].items():
            task_summaries.append({
                "task_id": task_id,
                "type": task_notes["task_type"],
                "summary": task_notes["summary"],
                "key_findings": task_notes["key_findings"]
            })
            
            # Aggregate statistics
            stats = task_notes.get("statistics", {})
            total_proteins += stats.get("total_proteins", 0)
            total_pathways += stats.get("pathways_represented", 0)
            total_annotations += stats.get("annotations_processed", 0)
            
            key_findings.extend(task_notes["key_findings"])
        
        summary = {
            "session_id": self.session_id,
            "query": self.current_notes["query"],
            "duration": self._calculate_duration(),
            "overall_statistics": {
                "total_proteins_discovered": total_proteins,
                "total_pathways_analyzed": total_pathways,
                "total_annotations_processed": total_annotations,
                "tasks_completed": len(self.current_notes["tasks"])
            },
            "key_findings": key_findings[:10],  # Top 10 findings
            "task_summaries": task_summaries,
            "recommendations": self._generate_recommendations()
        }
        
        self.current_notes["summary"] = summary
        return summary
    
    def _calculate_duration(self) -> str:
        """Calculate session duration"""
        if "started_at" not in self.current_notes:
            return "Unknown"
        
        start_time = datetime.fromisoformat(self.current_notes["started_at"])
        duration = datetime.now() - start_time
        return f"{duration.total_seconds():.1f} seconds"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on session data"""
        recommendations = []
        
        # Analyze task results for recommendations
        for task_id, task_notes in self.current_notes["tasks"].items():
            if task_notes["task_type"] == "pathway_discovery":
                proteins_found = task_notes.get("proteins_found", 0)
                if proteins_found > 20:
                    recommendations.append("Consider filtering results by specific pathways or functions")
                elif proteins_found < 5:
                    recommendations.append("Consider broadening search terms or including related pathways")
        
        if not recommendations:
            recommendations.append("Results look comprehensive - ready for detailed analysis")
        
        return recommendations
    
    def save_session(self) -> str:
        """Save session notes to file"""
        if not self.session_id:
            raise ValueError("No active session to save")
        
        # Generate final summary
        self.generate_session_summary()
        
        # Save to file
        notes_file = self.notes_dir / f"{self.session_id}.json"
        with open(notes_file, 'w') as f:
            json.dump(self.current_notes, f, indent=2)
        
        logger.info(f"📝 Saved session notes to {notes_file}")
        return str(notes_file)
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Load session notes from file"""
        notes_file = self.notes_dir / f"{session_id}.json"
        if not notes_file.exists():
            raise FileNotFoundError(f"Session notes not found: {session_id}")
        
        with open(notes_file, 'r') as f:
            self.current_notes = json.load(f)
        
        self.session_id = session_id
        logger.info(f"📝 Loaded session notes: {session_id}")
        return self.current_notes
    
    def get_summary_for_llm(self) -> str:
        """Get a formatted summary optimized for LLM processing"""
        summary = self.generate_session_summary()
        
        # Format for LLM consumption
        formatted = f"""
# Analysis Session Summary

**Query**: {summary['query']}
**Duration**: {summary['duration']}

## Overall Results
- **{summary['overall_statistics']['total_proteins_discovered']} proteins discovered**
- **{summary['overall_statistics']['total_pathways_analyzed']} pathways analyzed**
- **{summary['overall_statistics']['tasks_completed']} tasks completed**

## Key Findings
{chr(10).join(f"- {finding}" for finding in summary['key_findings'])}

## Task Breakdown
{chr(10).join(f"- **{task['task_id']}** ({task['type']}): {task['summary']}" for task in summary['task_summaries'])}

## Recommendations
{chr(10).join(f"- {rec}" for rec in summary['recommendations'])}
"""
        return formatted.strip()

# Global instance for easy access
task_notes = TaskNotesManager()