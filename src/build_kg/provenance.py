"""
Provenance Tracking Utilities
Functions for tracking data provenance and pipeline activities.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActivityEntity(BaseModel):
    """
    Represents a pipeline activity for provenance tracking.
    Maps to PROV-O Activity class.
    
    TODO: Add complete PROV-O compliance
    """
    id: str = Field(..., description="Unique activity identifier")
    name: str = Field(..., description="Activity name")
    description: Optional[str] = Field(None, description="Activity description")
    started_at: datetime = Field(..., description="Activity start time")
    ended_at: Optional[datetime] = Field(None, description="Activity end time")
    
    # Software and parameters
    software_name: str = Field(..., description="Software used")
    software_version: Optional[str] = Field(None, description="Software version")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    
    # Input/output entities
    used_entities: List[str] = Field(default_factory=list, description="Input entity IDs")
    generated_entities: List[str] = Field(default_factory=list, description="Output entity IDs")


class ProvenanceTracker:
    """
    Tracks provenance information for pipeline activities.
    
    TODO: Implement complete provenance tracking
    """
    
    def __init__(self, output_path: Optional[Path] = None):
        self.activities: List[ActivityEntity] = []
        self.output_path = output_path or Path("data/kg/provenance.json")
    
    def start_activity(
        self,
        activity_id: str,
        name: str,
        software_name: str,
        description: Optional[str] = None,
        software_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ActivityEntity:
        """
        Start tracking a new pipeline activity.
        
        TODO: Add activity validation and conflict detection
        """
        activity = ActivityEntity(
            id=activity_id,
            name=name,
            description=description,
            started_at=datetime.now(),
            software_name=software_name,
            software_version=software_version,
            parameters=parameters or {}
        )
        
        self.activities.append(activity)
        logger.info(f"Started activity: {activity_id}")
        return activity
    
    def end_activity(
        self,
        activity_id: str,
        generated_entities: Optional[List[str]] = None
    ) -> None:
        """
        Mark an activity as completed.
        
        TODO: Add activity lookup and validation
        """
        # Find activity by ID
        activity = None
        for a in self.activities:
            if a.id == activity_id:
                activity = a
                break
        
        if activity:
            activity.ended_at = datetime.now()
            if generated_entities:
                activity.generated_entities.extend(generated_entities)
            logger.info(f"Ended activity: {activity_id}")
        else:
            logger.warning(f"Activity not found: {activity_id}")
    
    def record_usage(
        self,
        activity_id: str,
        used_entities: List[str]
    ) -> None:
        """
        Record input entities used by an activity.
        
        TODO: Add entity validation
        """
        # Find and update activity
        for activity in self.activities:
            if activity.id == activity_id:
                activity.used_entities.extend(used_entities)
                logger.debug(f"Recorded usage for {activity_id}: {used_entities}")
                break
    
    def export_provenance(self, format: str = "json") -> None:
        """
        Export provenance information to file.
        
        TODO: Support multiple export formats (JSON, RDF, PROV-N)
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            import json
            provenance_data = {
                "activities": [activity.dict() for activity in self.activities],
                "exported_at": datetime.now().isoformat()
            }
            
            with open(self.output_path, 'w') as f:
                json.dump(provenance_data, f, indent=2, default=str)
            
            logger.info(f"Exported provenance to {self.output_path}")
        else:
            logger.warning(f"Export format not supported: {format}")


def create_pipeline_provenance(
    pipeline_run_id: str,
    input_genomes: List[str],
    output_dir: Path
) -> ProvenanceTracker:
    """
    Create provenance tracker for a complete pipeline run.
    
    TODO: Implement pipeline-level provenance setup
    
    Args:
        pipeline_run_id: Unique identifier for this pipeline run
        input_genomes: List of input genome identifiers
        output_dir: Directory where outputs will be written
        
    Returns:
        ProvenanceTracker instance configured for the pipeline
    """
    provenance_path = output_dir / "provenance.json"
    tracker = ProvenanceTracker(provenance_path)
    
    # TODO: Add pipeline-level activity
    tracker.start_activity(
        activity_id=f"pipeline_run_{pipeline_run_id}",
        name="Genome-to-LLM KG Pipeline",
        software_name="genome-kg",
        description="Complete genomic processing pipeline",
        parameters={
            "input_genomes": input_genomes,
            "output_directory": str(output_dir)
        }
    )
    
    logger.info(f"Created provenance tracker for run: {pipeline_run_id}")
    return tracker


def add_software_agent(
    activity: ActivityEntity,
    software_name: str,
    version: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add software agent information to an activity.
    
    TODO: Implement PROV-O agent modeling
    """
    # TODO: Create software agent entity
    # TODO: Link agent to activity with wasAssociatedWith
    logger.debug(f"Added software agent: {software_name}")
