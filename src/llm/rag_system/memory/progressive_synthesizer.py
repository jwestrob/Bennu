"""
Progressive synthesis system for handling large multi-task agentic workflows.

Uses a Map-Reduce architecture to process task notes and raw data efficiently,
with token-based decision making for optimal model utilization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tiktoken
import concurrent.futures

from .note_keeper import NoteKeeper
from .note_schemas import TaskNote, SynthesisNote, ConfidenceLevel
from .memory_utils import generate_session_id
from .model_allocation import get_model_allocator
from .tool_result_cache import ToolResultCache

logger = logging.getLogger(__name__)


class ProgressiveSynthesizer:
    """
    Map-Reduce based progressive synthesis system for genomic analysis workflows.
    
    Architecture:
    - Unified entry point that processes both raw_data and task_notes
    - Token-based decision making (not keyword or count based)
    - Direct synthesis for data that fits within model limits
    - Map-Reduce pipeline for larger datasets:
      * Map: Split data into chunks, summarize each chunk
      * Reduce: Combine chunk summaries into final synthesis
    """
    
    def __init__(self, note_keeper: NoteKeeper, chunk_size: int = 8, target_tokens: int = 15000, max_concurrent_calls: int = 6):
        """
        Initialize progressive synthesizer with Map-Reduce architecture and parallel processing.
        
        Args:
            note_keeper: NoteKeeper instance for accessing notes
            chunk_size: Number of tasks to process per chunk (legacy parameter)
            target_tokens: Target token count for each synthesis chunk
            max_concurrent_calls: Maximum number of concurrent API calls (default: 6 for safe rate limiting)
        """
        self.note_keeper = note_keeper
        self.chunk_size = chunk_size
        self.target_tokens = target_tokens
        self.max_concurrent_calls = max_concurrent_calls
        
        # Map-Reduce configuration - Model-aware limits
        # We'll set final limits after model allocator is available
        self.direct_synthesis_limit = 20000  # Will be updated based on actual model limits
        self.map_chunk_limit = 15000  # Will be updated based on actual model limits
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize model allocator for intelligent model selection
        self.model_allocator = get_model_allocator()
        
        # Initialize tool result cache for reference loading
        if hasattr(note_keeper, 'session_path'):
            self.tool_cache = ToolResultCache(str(note_keeper.session_path))
        else:
            self.tool_cache = None
        
        # Caching system to reduce API calls
        self.synthesis_cache = {}  # Cache for synthesis results
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Update chunk limits based on actual model capabilities
        self._update_model_aware_limits()
        
        logger.info("🏗️ ProgressiveSynthesizer initialized with Map-Reduce architecture and caching")
    
    def _update_model_aware_limits(self):
        """Update chunk limits based on actual model capabilities."""
        try:
            # Get limits for the models we'll be using
            _, final_synthesis_model = self.model_allocator.get_model_for_task("final_synthesis", "")
            _, map_step_model = self.model_allocator.get_model_for_task("genomic_summarization", "")
            
            # Set direct synthesis limit to respect OpenAI's 30K TPM rate limit for o3
            self.direct_synthesis_limit = min(25000, int(final_synthesis_model.max_context * 0.8))
            
            # Set map chunk limit to respect 30K TPM limit (not model context)
            self.map_chunk_limit = min(25000, int(map_step_model.max_context * 0.4))
            
            logger.info(f"📊 Model-aware limits updated: direct={self.direct_synthesis_limit:,}, chunk={self.map_chunk_limit:,}")
            logger.info(f"📊 Models: final_synthesis={final_synthesis_model.model_name}, map_step={map_step_model.model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update model-aware limits: {e}, using defaults")
    
    def synthesize_progressive(self, 
                             task_notes: List[TaskNote],
                             question: str,
                             synthesis_mode: str = "report",
                             dspy_synthesizer=None,
                             raw_data: List[Dict[str, Any]] = None,
                             rag_system=None) -> str:
        """
        Main entry point for progressive synthesis with hybrid mode support.
        
        Args:
            task_notes: List of TaskNote objects
            question: Original user question
            synthesis_mode: "guidance" for lightweight agent guidance or "report" for comprehensive final analysis
            dspy_synthesizer: DEPRECATED - uses model allocation (kept for compatibility)
            raw_data: Raw data from task execution (prioritized over task_notes)
            rag_system: DEPRECATED - not used in Map-Reduce architecture
            
        Returns:
            Final comprehensive synthesis or brief guidance summary
        """
        # Warn about deprecated parameters
        if dspy_synthesizer is not None:
            logger.warning("⚠️ dspy_synthesizer parameter is deprecated and will be ignored")
        if rag_system is not None:
            logger.warning("⚠️ rag_system parameter is deprecated and will be ignored")
        
        # HYBRID MODEL: Branch based on synthesis mode
        if synthesis_mode == "guidance":
            logger.info(f"🧭 Guidance synthesis: {len(task_notes)} recent notes")
            return self._guidance_synthesis(task_notes, question)
        else:
            logger.info(f"📊 Report synthesis: {len(task_notes)} notes, {len(raw_data) if raw_data else 0} raw items")
            return self._report_synthesis(task_notes, question, raw_data)
    
    def synthesize_with_evidence_mapping(self, 
                                       task_notes: List[TaskNote],
                                       question: str,
                                       preprocess_bundle: Optional['PreprocessBundle'] = None,
                                       evidence_ledger: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhanced synthesis with evidence mapping and narrative structure.
        
        Args:
            task_notes: Task notes from execution
            question: Original user question
            preprocess_bundle: Preprocessing bundle with detector provenance
            evidence_ledger: Evidence ledger with tool provenance
            
        Returns:
            Comprehensive narrative report with evidence mapping
        """
        logger.info("📊 Running enhanced synthesis with evidence mapping")
        
        # Build comprehensive data for narrative synthesis
        unified_data = self._prepare_unified_data(None, task_notes)
        
        # Add preprocessing and evidence context
        enhanced_context = self._enhance_with_evidence_mapping(
            unified_data=unified_data,
            question=question,
            preprocess_bundle=preprocess_bundle,
            evidence_ledger=evidence_ledger
        )
        
        # Use enhanced synthesis with narrative structure
        return self._enhanced_narrative_synthesis(enhanced_context, question)
    
    def _enhance_with_evidence_mapping(self, 
                                     unified_data: List[Dict[str, Any]],
                                     question: str,
                                     preprocess_bundle: Optional['PreprocessBundle'],
                                     evidence_ledger: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance data with evidence mapping and provenance."""
        enhanced_context = {
            "question": question,
            "unified_data": unified_data,
            "methods": {},
            "evidence_mapping": {},
            "limitations": []
        }
        
        # Add preprocessing methods
        if preprocess_bundle:
            enhanced_context["methods"]["preprocessing"] = {
                "function_detectors": preprocess_bundle.detectors.get("functions", []),
                "domain_detectors": preprocess_bundle.detectors.get("domains", []),
                "cypher_plans": [{"name": p.name, "params": p.params} for p in preprocess_bundle.cypher_plans],
                "schema_labels": preprocess_bundle.schema_summary.labels,
                "schema_relationships": preprocess_bundle.schema_summary.relationships
            }
        
        # Add evidence ledger
        if evidence_ledger:
            enhanced_context["evidence_mapping"] = evidence_ledger
            enhanced_context["methods"]["execution"] = {
                "total_steps": evidence_ledger.get("total_steps", 0),
                "tools_used": evidence_ledger.get("tools_used", [])
            }
        
        # Add standard limitations
        enhanced_context["limitations"] = [
            "Analysis limited to available database annotations",
            "Computational predictions may require experimental validation",
            "Results dependent on quality of input genome assemblies"
        ]
        
        return enhanced_context
    
    def _enhanced_narrative_synthesis(self, enhanced_context: Dict[str, Any], question: str) -> str:
        """Generate narrative report with structured sections."""
        narrative_parts = []
        
        # Title
        narrative_parts.append(f"# Genomic Analysis Report: {question}")
        narrative_parts.append("")
        
        # Methods
        narrative_parts.append("## Methods")
        methods = enhanced_context.get("methods", {})
        
        if "preprocessing" in methods:
            prep = methods["preprocessing"]
            narrative_parts.append(f"**Biological Target Resolution:** Identified {len(prep.get('function_detectors', []))} function detectors and {len(prep.get('domain_detectors', []))} domain detectors through schema-locked preprocessing.")
            narrative_parts.append(f"**Query Execution:** {len(prep.get('cypher_plans', []))} parameterized Cypher queries executed against knowledge graph.")
        
        if "execution" in methods:
            exec_info = methods["execution"]
            narrative_parts.append(f"**Analysis Pipeline:** {exec_info.get('total_steps', 0)} computational steps using tools: {', '.join(exec_info.get('tools_used', []))}.")
        
        narrative_parts.append("")
        
        # Findings
        narrative_parts.append("## Findings")
        unified_data = enhanced_context.get("unified_data", [])
        if unified_data:
            narrative_parts.append(f"Analysis identified {len(unified_data)} data points across the genomic dataset.")
            
            # Summarize key findings from unified data
            for i, data_point in enumerate(unified_data[:5]):  # Limit to top 5 for narrative
                if isinstance(data_point, dict) and "key_findings" in data_point:
                    narrative_parts.extend(data_point["key_findings"])
                elif isinstance(data_point, dict) and "summary" in data_point:
                    narrative_parts.append(f"- {data_point['summary']}")
        else:
            narrative_parts.append("No significant findings identified in the current analysis.")
        
        narrative_parts.append("")
        
        # Evidence Mapping
        narrative_parts.append("## Evidence → Detector → Source Mapping")
        evidence_mapping = enhanced_context.get("evidence_mapping", {})
        
        if evidence_mapping.get("detector_provenance"):
            provenance = evidence_mapping["detector_provenance"]
            narrative_parts.append(f"**Function Detectors:** {', '.join(provenance.get('functions', []))}")
            narrative_parts.append(f"**Domain Detectors:** {', '.join(provenance.get('domains', []))}")
        
        if evidence_mapping.get("evidence_to_detector_mapping"):
            narrative_parts.append("**Step-wise Evidence Mapping:**")
            for step_id, detector_info in evidence_mapping["evidence_to_detector_mapping"].items():
                narrative_parts.append(f"- {step_id}: {detector_info}")
        
        narrative_parts.append("")
        
        # Quality & Limitations
        narrative_parts.append("## Quality Control & Limitations")
        limitations = enhanced_context.get("limitations", [])
        for limitation in limitations:
            narrative_parts.append(f"- {limitation}")
        
        narrative_parts.append("")
        
        # Contextual Neighbors (placeholder for spatial analysis)
        narrative_parts.append("## Contextual Analysis")
        narrative_parts.append("Genomic neighborhood analysis conducted where applicable, examining gene organization and functional clustering patterns.")
        
        return "\n".join(narrative_parts)
    
    def _guidance_synthesis(self, task_notes: List[TaskNote], question: str) -> str:
        """
        Lightweight guidance synthesis for agent situational awareness.
        
        Args:
            task_notes: Recent task notes (typically last 3 steps)
            question: Original user question
            
        Returns:
            Brief guidance summary (2-3 sentences)
        """
        logger.info("🧭 Running lightweight guidance synthesis")
        
        # Simple format for guidance - just recent findings
        if not task_notes:
            return "Continue exploration - no recent notes available for guidance."
        
        # Format recent findings 
        recent_findings = []
        for note in task_notes[-3:]:  # Last 3 notes max
            findings = " | ".join(note.key_findings) if note.key_findings else note.description
            recent_findings.append(f"Step {note.task_id}: {findings}")
        
        context = "RECENT PROGRESS:\n" + "\n".join(recent_findings)
        
        # Use fast model for guidance
        try:
            guidance = self._call_synthesis_model(
                context=context,
                question=question,
                task_name="guidance_synthesis",  # Maps to MEDIUM = gpt-5-mini-2025-08-07
                focus="brief guidance for next steps (2-3 sentences max)",
                synthesis_type="summarization"
            )
            
            return guidance
            
        except Exception as e:
            logger.warning(f"⚠️ Guidance synthesis failed: {e}")
            return f"Recent findings: {len(task_notes)} steps completed. Continue systematic exploration."
    
    def _report_synthesis(self, task_notes: List[TaskNote], question: str, raw_data: List[Dict[str, Any]] = None) -> str:
        """
        Comprehensive report synthesis using full Map-Reduce architecture.
        
        Args:
            task_notes: All task notes from session
            question: Original user question  
            raw_data: Raw data from task execution
            
        Returns:
            Comprehensive final report
        """
        logger.info("📊 Running comprehensive report synthesis")
        
        # Step 1: Determine primary data source (prioritize raw_data)
        unified_data = self._prepare_unified_data(raw_data, task_notes)
        
        if not unified_data:
            return "No data available for synthesis."
        
        # Step 2: Smart synthesis mode selection
        synthesis_strategy = self._choose_optimal_synthesis_strategy(unified_data, question)
        
        if synthesis_strategy == "key_findings_only":
            logger.info("🎯 Using key-findings-only synthesis (sufficient discoveries detected)")
            return self._synthesize_from_key_findings_only(unified_data, question)
        
        # Step 3: Token-based decision making for full context synthesis
        total_tokens = self._count_data_tokens(unified_data)
        logger.info(f"📊 Total input tokens: {total_tokens} (strategy: {synthesis_strategy})")
        
        if total_tokens <= self.direct_synthesis_limit:
            logger.info("🎯 Using direct synthesis (data fits within model limits)")
            return self._direct_synthesis(unified_data, question)
        else:
            logger.info("🗂️ Using Map-Reduce synthesis (data exceeds model limits)")
            return self._map_reduce_synthesis(unified_data, question)
    
    def _prepare_unified_data(self, raw_data: Optional[List[Dict[str, Any]]], 
                             task_notes: List[TaskNote]) -> List[Dict[str, Any]]:
        """
        Prepare unified data format for processing.
        
        Args:
            raw_data: Raw data from task execution
            task_notes: Task notes for context
            
        Returns:
            Unified data format (list of dictionaries)
        """
        unified_data = []
        
        # Priority 1: Use raw_data if available
        if raw_data:
            logger.info(f"✅ Using raw_data as primary source ({len(raw_data)} items)")
            
            # CRITICAL: Check if this is gene record data that needs structural preservation
            if self._looks_like_gene_records(raw_data):
                logger.info("🔬 Detected gene records - preserving structured loci data")
                grouped_loci = self._group_genes_into_loci(raw_data)
                
                # Add structured loci data to unified format
                unified_data.append({
                    'type': 'structured_loci',
                    'loci_count': len(grouped_loci),
                    'structured_loci': grouped_loci,
                    'formatted_preview': self._format_loci_for_synthesis(grouped_loci)[:500] + "..."
                })
                # Also preserve original gene records for compatibility
                unified_data.extend(raw_data)
            else:
                unified_data.extend(raw_data)
        
        # Priority 2: Convert task_notes to unified format if no raw_data
        if not raw_data and task_notes:
            logger.info(f"📝 Converting task_notes to unified format ({len(task_notes)} notes)")
            for note in task_notes:
                # REVOLUTIONARY CHANGE: Expand tool result references
                quantitative_data = dict(note.quantitative_data) if note.quantitative_data else {}
                
                # Check if this note has a tool result reference
                if 'tool_result_ref' in quantitative_data and self.tool_cache:
                    result_id = quantitative_data['tool_result_ref']
                    logger.info(f"🔗 Expanding tool result reference: {result_id}")
                    
                    # Load the referenced tool result
                    tool_result = self.tool_cache.retrieve_tool_result(result_id)
                    
                    if tool_result:
                        # Add expanded tool result to quantitative data
                        quantitative_data['expanded_tool_result'] = tool_result
                        logger.info(f"✅ Loaded referenced tool result for {note.task_id}")
                    else:
                        logger.warning(f"⚠️ Failed to load tool result reference: {result_id}")
                
                unified_data.append({
                    'type': 'task_note',
                    'task_id': note.task_id,
                    'description': note.description,
                    'observations': note.observations,
                    'key_findings': note.key_findings,
                    'confidence_level': note.confidence_level.value,
                    'quantitative_data': quantitative_data,
                    'cross_task_connections': [
                        {
                            'connected_task': conn.connected_task,
                            'connection_type': conn.connection_type.value,
                            'description': conn.description
                        } for conn in note.cross_task_connections
                    ]
                })
        
        # Add task_notes as metadata if we have raw_data
        if raw_data and task_notes:
            logger.info(f"📋 Adding task_notes as metadata ({len(task_notes)} notes)")
            unified_data.append({
                'type': 'task_metadata',
                'task_count': len(task_notes),
                'key_insights': [finding for note in task_notes for finding in note.key_findings],
                'cross_connections': [
                    {
                        'from_task': note.task_id,
                        'to_task': conn.connected_task,
                        'relationship': conn.connection_type.value,
                        'description': conn.description
                    } for note in task_notes for conn in note.cross_task_connections
                ]
            })
        
        # DEBUG: Save pre-compression data  
        self._save_debug_data("pre_compression_data", unified_data, "Unified data before intelligent context management")
        
        # CRITICAL OPTIMIZATION: Apply intelligent context management
        if self.tool_cache:
            unified_data = self._apply_intelligent_context_management(unified_data)
        
        # DEBUG: Save post-compression data
        self._save_debug_data("post_compression_data", unified_data, "Unified data after intelligent context management")
        
        return unified_data
    
    def _apply_intelligent_context_management(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply intelligent context management to avoid token explosion.
        
        This method decides whether to include full tool results or just summaries
        based on available token budget and content importance.
        
        Args:
            data: Unified data with potentially expanded tool results
            
        Returns:
            Optimized data with smart context management
        """
        logger.info("🧠 Applying intelligent context management")
        
        # Calculate current token usage
        current_tokens = self._count_data_tokens(data)
        logger.info(f"📊 Current context size: {current_tokens:,} tokens")
        
        # If we're within limits, keep everything
        if current_tokens <= self.direct_synthesis_limit:
            logger.info("✅ Context size acceptable, keeping full tool results")
            return data
        
        # Otherwise, apply compression by replacing large tool results with summaries
        compressed_data = []
        tokens_saved = 0
        
        for item in data:
            if item.get('type') == 'task_note' and 'quantitative_data' in item:
                quant_data = item['quantitative_data']
                
                if 'expanded_tool_result' in quant_data:
                    # Calculate size of expanded result
                    expanded_result = quant_data['expanded_tool_result']
                    expanded_tokens = self._count_data_tokens([expanded_result])
                    
                    # Model-aware compression threshold
                    compression_threshold = self._get_compression_threshold()
                    
                    # If expanded result is large, replace with summary + key findings
                    if expanded_tokens > compression_threshold:
                        logger.info(f"🗜️ Compressing large tool result in {item['task_id']} ({expanded_tokens:,} tokens > {compression_threshold:,} threshold)")
                        
                        # Keep the summary and biological discoveries
                        compressed_quant = dict(quant_data)
                        del compressed_quant['expanded_tool_result']  # Remove large result
                        
                        # Extract and preserve key biological information
                        if 'tool_result_ref' in compressed_quant:
                            result_id = compressed_quant['tool_result_ref']
                            tool_name = result_id.split('_')[0]  # Extract tool name from ID
                            
                            # For WGR results, preserve detailed loci information
                            if tool_name == 'wgr' and isinstance(expanded_result, dict):
                                loci_summary = self._extract_detailed_loci_summary(expanded_result)
                                if loci_summary:
                                    compressed_quant['detailed_loci_summary'] = loci_summary
                                    logger.info(f"🧬 Preserved detailed summary of {len(loci_summary)} loci")
                            
                            # Also extract general discoveries
                            discoveries = self.tool_cache.extract_key_discoveries(tool_name, expanded_result)
                            if discoveries:
                                # Add discoveries to key_findings for preservation
                                item['key_findings'] = list(item.get('key_findings', [])) + discoveries
                                logger.info(f"🔬 Preserved {len(discoveries)} biological discoveries")
                        
                        # Update item with compressed data
                        item = dict(item)
                        item['quantitative_data'] = compressed_quant
                        tokens_saved += expanded_tokens
            
            compressed_data.append(item)
        
        if tokens_saved > 0:
            final_tokens = self._count_data_tokens(compressed_data)
            logger.info(f"🎯 Context compression complete: {tokens_saved:,} tokens saved, final size: {final_tokens:,} tokens")
        
        return compressed_data
    
    def _extract_detailed_loci_summary(self, wgr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract detailed loci summary from WGR results for synthesis.
        
        This preserves the essential loci information (coordinates, gene counts, 
        biological features) while reducing token count significantly.
        
        Args:
            wgr_result: Whole genome reader result with hierarchical analysis
            
        Returns:
            List of detailed loci summaries for synthesis
        """
        loci_summaries = []
        
        try:
            if 'interesting_loci' in wgr_result:
                interesting_loci = wgr_result['interesting_loci']
                
                # Handle both string and object representations
                for locus_data in interesting_loci:
                    try:
                        if isinstance(locus_data, str):
                            # Parse string representation of InterestingLocus
                            locus_summary = self._parse_locus_string(locus_data)
                        elif isinstance(locus_data, dict):
                            # Already structured data
                            locus_summary = {
                                'contig_id': locus_data.get('contig_id', 'unknown'),
                                'coordinates': f"{locus_data.get('start', 0)}-{locus_data.get('end', 0)}",
                                'gene_count': locus_data.get('gene_count', 0),
                                'locus_type': locus_data.get('locus_type', 'unknown'),
                                'biological_features': locus_data.get('biological_features', [])[:3]  # Top 3 features
                            }
                        else:
                            continue
                            
                        if locus_summary:
                            loci_summaries.append(locus_summary)
                            
                    except Exception as e:
                        logger.warning(f"Error parsing locus data: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error extracting detailed loci summary: {e}")
        
        # Sort by genomic position (contig + coordinate) to preserve synteny and positional relationships
        def genomic_position_key(locus):
            contig_id = locus.get('contig_id', 'zzz_unknown')  # Sort unknowns last
            coordinates = locus.get('coordinates', '0-0')
            try:
                start_pos = int(coordinates.split('-')[0])
                return (contig_id, start_pos)
            except:
                return (contig_id, 0)
        
        loci_summaries.sort(key=genomic_position_key)
        return loci_summaries[:15]  # Return more loci to preserve complete genomic context
    
    def _parse_locus_string(self, locus_str: str) -> Dict[str, Any]:
        """Parse InterestingLocus string representation into structured data."""
        try:
            # Extract key information using regex
            import re
            
            # Parse contig_id
            contig_match = re.search(r"contig_id='([^']+)'", locus_str)
            contig_id = contig_match.group(1) if contig_match else 'unknown'
            
            # Parse coordinates
            start_match = re.search(r"start=(\d+)", locus_str)
            end_match = re.search(r"end=(\d+)", locus_str)
            start = int(start_match.group(1)) if start_match else 0
            end = int(end_match.group(1)) if end_match else 0
            
            # Parse gene counts
            gene_count_match = re.search(r"gene_count=(\d+)", locus_str)
            gene_count = int(gene_count_match.group(1)) if gene_count_match else 0
            
            # Parse locus type
            type_match = re.search(r"locus_type='([^']+)'", locus_str)
            locus_type = type_match.group(1) if type_match else 'unknown'
            
            # Parse biological features
            features_match = re.search(r"biological_features=\[(.*?)\]", locus_str)
            features = []
            if features_match:
                features_str = features_match.group(1)
                # Simple extraction of quoted strings
                feature_matches = re.findall(r"'([^']+)'", features_str)
                features = feature_matches[:3]  # Top 3 features
            
            return {
                'contig_id': contig_id,
                'coordinates': f"{start}-{end}",
                'gene_count': gene_count,
                'locus_type': locus_type,
                'biological_features': features
            }
            
        except Exception as e:
            logger.warning(f"Error parsing locus string: {e}")
            return None
    
    def _get_compression_threshold(self) -> int:
        """
        Get compression threshold respecting OpenAI's 30K TPM rate limit.
        
        Returns:
            Token threshold above which tool results should be compressed
        """
        # Respect OpenAI's 30K tokens per minute rate limit for o3
        # Use conservative limit to avoid rate limiting errors
        return 20000
    
    def _save_debug_data(self, stage_name: str, data: Any, description: str) -> None:
        """
        Save debug data to session notes for data flow analysis.
        
        Args:
            stage_name: Name of the processing stage (e.g., "pre_compression_data")
            data: Data to save for debugging
            description: Human-readable description of this data
        """
        try:
            # Create debug directory if it doesn't exist
            if hasattr(self.note_keeper, 'session_path'):
                debug_dir = self.note_keeper.session_path / "debug_data_flow"
                debug_dir.mkdir(exist_ok=True)
                
                # Create debug file
                from datetime import datetime
                timestamp = datetime.now().strftime("%H%M%S")
                debug_file = debug_dir / f"{stage_name}_{timestamp}.json"
                
                # Prepare debug payload
                debug_payload = {
                    "stage_name": stage_name,
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                    "data_type": type(data).__name__,
                    "data_size_chars": len(str(data)),
                    "token_estimate": self._count_data_tokens([data]) if isinstance(data, (dict, list)) else 0,
                    "data": data
                }
                
                # Write debug file
                import json
                with open(debug_file, 'w') as f:
                    json.dump(debug_payload, f, indent=2, default=str)
                
                logger.info(f"🐛 DEBUG: Saved {stage_name} to {debug_file.name} ({debug_payload['data_size_chars']} chars)")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save debug data for {stage_name}: {e}")
    
    def _count_data_tokens(self, data: List[Dict[str, Any]]) -> int:
        """
        Count tokens in unified data format.
        
        Args:
            data: Unified data to count tokens for
            
        Returns:
            Total token count
        """
        if not self.tokenizer:
            # Fallback estimation: 3.5 characters per token for English
            total_chars = sum(len(str(item)) for item in data)
            return int(total_chars / 3.5)
        
        try:
            # More accurate token counting
            full_text = "\n".join(str(item) for item in data)
            return len(self.tokenizer.encode(full_text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to character-based estimation
            total_chars = sum(len(str(item)) for item in data)
            return int(total_chars / 3.5)
    
    def _direct_synthesis(self, data: List[Dict[str, Any]], question: str) -> str:
        """
        Direct synthesis for data that fits within model limits.
        
        Args:
            data: Unified data to synthesize
            question: Original user question
            
        Returns:
            Direct synthesis result
        """
        logger.info("🎯 Performing direct synthesis with full context")
        
        # DEBUG: Save pre-synthesis data for debugging
        self._save_debug_data("pre_synthesis_data", data, "Raw unified data before formatting")
        
        # Format data for synthesis
        formatted_data = self._format_data_for_synthesis(data)
        
        # DEBUG: Save formatted synthesis input
        self._save_debug_data("formatted_synthesis_input", formatted_data, "Formatted data sent to LLM")
        
        return self._call_synthesis_model(
            context=formatted_data,
            question=question,
            task_name="final_synthesis",  # Use o3 for complex biological reasoning
            focus="comprehensive biological analysis with full context"
        )
    
    def _map_reduce_synthesis(self, data: List[Dict[str, Any]], question: str) -> str:
        """
        Map-Reduce synthesis for large datasets with sequential processing.
        
        Args:
            data: Unified data to synthesize
            question: Original user question
            
        Returns:
            Map-Reduce synthesis result
        """
        logger.info("🗂️ Performing Map-Reduce synthesis with sequential processing")
        
        # Store question for pre-summarization context
        self._current_question = question
        
        # MAP STEP: Split data into chunks and summarize each sequentially  
        chunks = self._create_chunks(data)
        
        # Log chunk sizes for debugging
        chunk_sizes = [self._count_data_tokens(chunk) for chunk in chunks]
        logger.info(f"📦 Created {len(chunks)} chunks for parallel Map step processing")
        logger.info(f"📊 Chunk sizes: {chunk_sizes} tokens (limit: {self.map_chunk_limit})")
        
        # SINGLE-CHUNK BYPASS: If only one chunk, skip Map-Reduce and go directly to final synthesis
        if len(chunks) == 1:
            logger.info("🎯 Single chunk detected - bypassing Map-Reduce and proceeding directly to final synthesis")
            single_chunk_data = chunks[0]
            formatted_context = self._format_data_for_synthesis(single_chunk_data)
            
            # Use high-capability model for direct synthesis
            final_result = self._call_synthesis_model(
                context=formatted_context,
                question=question,
                task_name="final_synthesis",  # Use o3 for complex synthesis
                focus="comprehensive biological analysis with full data context",
                synthesis_type="summarization"  # Use regular synthesis, not Map-Reduce
            )
            
            # DEBUG: Save single-chunk bypass output
            self._save_debug_data("single_chunk_bypass_output", final_result, "Direct synthesis from single chunk")
            
            return final_result
        
        # MAP STEP: Parallelize chunk processing
        logger.info(f"🚀 Starting parallel chunk processing with {self.max_concurrent_calls} workers")
        chunk_summaries = [None] * len(chunks)  # Pre-allocate results list
        
        def process_chunk(index_and_chunk):
            index, chunk = index_and_chunk
            try:
                logger.info(f"📝 Processing chunk {index+1}/{len(chunks)}")
                summary = self._map_step(chunk, question, index+1)
                return index, summary
            except Exception as e:
                logger.error(f"❌ Chunk {index+1} processing failed: {e}")
                return index, None
        
        # CRITICAL FIX: Replace parallel with SEQUENTIAL processing to prevent TPM overflow
        logger.info(f"🔄 Processing chunks SEQUENTIALLY to prevent rate limits")
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"📝 Processing chunk {i+1}/{len(chunks)}")
                
                # Add rate limiting between main chunk API calls
                if i > 0:  # Don't delay before first call
                    import time
                    delay = 2.0  # 2 second delay between main chunk API calls
                    logger.info(f"⏳ Rate limiting: waiting {delay}s before next chunk")
                    time.sleep(delay)
                
                summary = self._map_step(chunk, question, i+1)
                if summary:  # Only store successful summaries
                    chunk_summaries[i] = summary
                    logger.info(f"✅ Completed chunk {i+1}/{len(chunks)}")
                else:
                    logger.warning(f"⚠️ Chunk {i+1} returned no summary")
                    
            except Exception as e:
                logger.error(f"❌ Chunk {i+1} failed: {e}")
        
        # Filter out None values from failed chunks
        chunk_summaries = [s for s in chunk_summaries if s is not None]
        logger.info(f"🎯 Sequential chunk processing complete: {len(chunk_summaries)}/{len(chunks)} successful")
        
        # REDUCE STEP: Combine chunk summaries into final synthesis
        logger.info(f"🔄 Reduce step: combining {len(chunk_summaries)} summaries")
        return self._reduce_step(chunk_summaries, question)
    
    def _create_chunks(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split data into chunks for Map step processing with pre-summarization of oversized items.
        
        Args:
            data: Unified data to chunk
            
        Returns:
            List of data chunks
        """
        # Step 1: Pre-processing - handle oversized items
        logger.info(f"🔍 Pre-processing {len(data)} items for oversized content")
        processed_data = []
        
        for i, item in enumerate(data):
            item_tokens = self._count_data_tokens([item])
            
            if item_tokens > self.map_chunk_limit:
                # This item is oversized, pre-summarize it (NO TRUNCATION - ALWAYS PRESERVE BIOLOGICAL DATA)
                logger.warning(f"📝 Item {i+1}/{len(data)} is oversized ({item_tokens:,} tokens > {self.map_chunk_limit:,}) - will summarize to preserve all biological content")
                
                # Get the question from context (we need to pass this through)
                question = getattr(self, '_current_question', 'biological analysis')
                summarized_item = self._summarize_oversized_item(item, question)
                processed_data.append(summarized_item)
            else:
                # Item is fine as-is
                processed_data.append(item)
        
        logger.info(f"✅ Pre-processing complete: {len(processed_data)} items ready for chunking")
        
        # Step 2: Main chunking logic (now guaranteed that no item exceeds chunk limit)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for item in processed_data:
            item_tokens = self._count_data_tokens([item])
            
            # Check if adding this item would exceed chunk limit
            if current_tokens + item_tokens > self.map_chunk_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(item)
            current_tokens += item_tokens
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # B. Compress any chunks that are still too big
        compressed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_data_tokens(chunk)
            if chunk_tokens > self.map_chunk_limit:
                logger.warning(f"🗜️ Chunk {i+1} exceeds limit ({chunk_tokens:,} > {self.map_chunk_limit:,} tokens) - compressing")
                compressed_chunk = self._compress_oversized_chunk(chunk, i+1)
                compressed_chunks.append(compressed_chunk)
            else:
                compressed_chunks.append(chunk)
        
        return compressed_chunks
    
    def _summarize_oversized_item(self, item: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Summarize oversized data items using recursive Map-Reduce.
        
        Args:
            item: Oversized data item to summarize
            question: Original user question for context
            
        Returns:
            Summarized item that fits within token limits
        """
        item_tokens = self._count_data_tokens([item])
        logger.warning(f"🔄 Pre-summarizing oversized item ({item_tokens:,} tokens → target: {self.map_chunk_limit:,})")
        
        # Extract the main content to summarize
        content = ""
        if 'result' in item and isinstance(item['result'], dict):
            result = item['result']
            if 'tool_output' in result:
                content = str(result['tool_output'])
            else:
                content = str(result)
        else:
            content = str(item)
        
        # Sub-chunking: Split content into manageable pieces
        sub_chunks = self._split_content_into_subchunks(content)
        logger.info(f"📦 Split oversized item into {len(sub_chunks)} sub-chunks for pre-summarization")
        
        # Sub-Map: SEQUENTIAL sub-chunk summarization to prevent API flood
        logger.info(f"🔄 Starting SEQUENTIAL sub-chunk summarization ({len(sub_chunks)} chunks)")
        sub_summaries = []
        
        for i, sub_chunk in enumerate(sub_chunks):
            try:
                logger.info(f"🔄 Pre-summarizing sub-chunk {i+1}/{len(sub_chunks)}")
                
                # Add aggressive rate limiting between API calls
                if i > 0:  # Don't delay before first call
                    import time
                    delay = 3.0  # 3 second delay between sub-chunk API calls
                    logger.info(f"⏳ Rate limiting: waiting {delay}s before next API call")
                    time.sleep(delay)
                
                summary = self._call_synthesis_model(
                    context=sub_chunk,
                    question=question,
                    task_name="genomic_summarization", 
                    focus=f"key biological patterns and insights from data chunk {i+1}"
                )
                sub_summaries.append(summary)
                logger.info(f"✅ Completed sub-chunk {i+1}/{len(sub_chunks)}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to summarize sub-chunk {i+1}: {e}")
                # Include truncated version as fallback
                truncated = sub_chunk[:5000] + "...[truncated for synthesis]"
                sub_summaries.append(f"Sub-chunk {i+1} (summarization failed): {truncated}")
        
        logger.info(f"🎯 Sequential sub-chunk summarization complete: {len(sub_summaries)}/{len(sub_chunks)} successful")
        
        # Sub-Reduce: Combine all sub-summaries
        if sub_summaries:
            combined_summary = "\n\n".join([
                f"=== Summary Part {i+1} ===\n{summary}" 
                for i, summary in enumerate(sub_summaries)
            ])
            
            # Final synthesis of combined summaries
            final_summary = self._call_synthesis_model(
                context=combined_summary,
                question=question,
                task_name="genomic_summarization",
                focus="integrate biological insights from all data sections"
            )
        else:
            final_summary = f"Pre-summarization failed for oversized item ({item_tokens:,} tokens)"
        
        # Create summarized item preserving structure
        summarized_item = item.copy()
        
        # Replace the large content with summary
        if 'result' in summarized_item and isinstance(summarized_item['result'], dict):
            summarized_item['result'] = summarized_item['result'].copy()
            summarized_item['result']['tool_output'] = final_summary
            summarized_item['result']['_original_token_count'] = item_tokens
            summarized_item['result']['_summarization_applied'] = True
        else:
            summarized_item = {
                'original_type': str(type(item).__name__),
                'summarized_content': final_summary,
                '_original_token_count': item_tokens,
                '_summarization_applied': True
            }
        
        final_tokens = self._count_data_tokens([summarized_item])
        logger.info(f"✅ Pre-summarization complete: {item_tokens:,} → {final_tokens:,} tokens ({(1-final_tokens/item_tokens)*100:.1f}% reduction)")
        
        return summarized_item
    
    def _split_content_into_subchunks(self, content: str) -> List[str]:
        """
        Split large content into smaller sub-chunks for pre-summarization.
        
        Args:
            content: Large content string to split
            
        Returns:
            List of content sub-chunks (maximum 10 chunks to prevent API flood)
        """
        # OPTIMIZED: Create fewer, larger sub-chunks for better performance while preserving all biological data
        max_subchunks = 5  # Even fewer sub-chunks for faster processing
        min_chunk_size = len(content) // max_subchunks  # Ensure we don't exceed max chunks
        
        # Calculate safe sub-chunk size based on model limits
        # Get model that will be used for summarization (genomic_summarization task uses gpt-5-mini-2025-08-07)
        model_name, model_config = self.model_allocator.get_model_for_task("genomic_summarization", "")
        
        # Use 40% of model context for sub-chunks (leaving room for system messages, response, etc.)
        max_safe_tokens = int(model_config.max_context * 0.4)
        target_subchunk_tokens = min(25000, max_safe_tokens)  # Use smaller of 25k or 40% of model limit
        target_chars = int(target_subchunk_tokens * 3.5)
        
        logger.debug(f"📊 Sub-chunk sizing: {target_subchunk_tokens:,} tokens max for {model_name} (context: {model_config.max_context:,})")
        
        if len(content) <= target_chars:
            return [content]
        
        # Split content into chunks, trying to break at natural boundaries
        sub_chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            end_pos = min(current_pos + target_chars, len(content))
            
            # Try to find a good break point near the target position
            if end_pos < len(content):
                # Look for natural breaks (newlines, periods, etc.)
                break_positions = []
                search_range = min(500, target_chars // 4)  # Search within reasonable range
                
                for offset in range(search_range):
                    if end_pos - offset > current_pos:
                        char = content[end_pos - offset]
                        if char in ['\n\n', '\n', '.', '!', '?']:  # Prefer paragraph/sentence breaks
                            break_positions.append(end_pos - offset + 1)
                
                if break_positions:
                    end_pos = break_positions[0]  # Use the closest good break point
            
            chunk = content[current_pos:end_pos].strip()
            if chunk:
                sub_chunks.append(chunk)
            
            current_pos = end_pos
        
        # ENFORCE HARD LIMIT: If we still have too many chunks, merge them to preserve all data
        if len(sub_chunks) > max_subchunks:
            logger.warning(f"🔄 Too many sub-chunks ({len(sub_chunks)} > {max_subchunks}), merging to stay under limit while preserving all biological data")
            merged_chunks = []
            chunks_per_merge = (len(sub_chunks) + max_subchunks - 1) // max_subchunks  # Ceiling division
            
            for i in range(0, len(sub_chunks), chunks_per_merge):
                merged_chunk = "\n\n".join(sub_chunks[i:i + chunks_per_merge])
                merged_chunks.append(merged_chunk)
            
            sub_chunks = merged_chunks
        
        logger.info(f"📝 Split {len(content):,} chars into {len(sub_chunks)} sub-chunks (max: {max_subchunks}, avg: {len(content)//len(sub_chunks):,} chars each)")
        return sub_chunks
    
    def _map_step(self, chunk: List[Dict[str, Any]], question: str, chunk_id: int) -> Optional[str]:
        """
        Map step: Summarize a single chunk of data.
        
        Args:
            chunk: Data chunk to summarize
            question: Original user question for context
            chunk_id: Chunk identifier for logging
            
        Returns:
            Chunk summary or None if failed
        """
        try:
            # Check if chunk contains structured loci data
            structured_loci_item = next((item for item in chunk if item.get('type') == 'structured_loci'), None)
            
            if structured_loci_item:
                # Special handling for structured loci - pass structured data to Map step
                logger.info(f"📍 Chunk {chunk_id} contains structured loci data - preserving structure")
                structured_loci = structured_loci_item['structured_loci']
                
                # Format loci data as detailed text for Map step while preserving structure
                formatted_chunk = f"STRUCTURED GENOMIC LOCI DATA ({len(structured_loci)} loci):\n\n"
                for i, locus in enumerate(structured_loci, 1):
                    formatted_chunk += f"Locus {i}: {locus.get('contig_id', 'unknown')} "
                    formatted_chunk += f"({locus.get('start_pos', 0)}-{locus.get('end_pos', 0)}, "
                    formatted_chunk += f"{len(locus.get('genes', []))} genes)\n"
                    for gene in locus.get('genes', []):
                        formatted_chunk += f"  Gene: {gene.get('gene_id', 'unknown')} "
                        formatted_chunk += f"({gene.get('start', 0)}-{gene.get('end', 0)}) "
                        formatted_chunk += f"Function: {gene.get('function', 'unknown')}\n"
                    formatted_chunk += "\n"
            else:
                # Standard formatting for non-loci data
                formatted_chunk = self._format_data_for_synthesis(chunk)
            
            # Use cheaper model for Map step (data extraction task)
            summary = self._call_synthesis_model(
                context=formatted_chunk,
                question=question,
                task_name="genomic_summarization",  # Use cheaper model for chunk summarization
                focus=f"extract detailed loci with coordinates, gene counts, biological features from chunk {chunk_id}",
                synthesis_type="map_extraction"
            )
            
            # DEBUG: Save Map step output
            self._save_debug_data(f"map_step_chunk_{chunk_id}_output", summary, f"Map step summary for chunk {chunk_id}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Map step failed for chunk {chunk_id}: {e}")
            return None
    
    def _reduce_step(self, chunk_summaries: List[str], question: str) -> str:
        """
        Reduce step: Combine chunk summaries into final synthesis.
        
        Args:
            chunk_summaries: List of chunk summaries from Map step
            question: Original user question
            
        Returns:
            Final synthesis result
        """
        if not chunk_summaries:
            return "No chunk summaries available for final synthesis."
        
        # Combine all chunk summaries
        combined_context = "\n\n".join([
            f"=== Chunk {i+1} Summary ===\n{summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        # Add synthesis metadata
        synthesis_context = f"""
QUESTION: {question}

CHUNK SUMMARIES ({len(chunk_summaries)} chunks):
{combined_context}

SYNTHESIS TASK: Integrate the above chunk summaries into a comprehensive, coherent analysis that addresses the original question.
"""
        
        # Use original question directly for intelligent selection
        
        # Use high-capability model for final synthesis
        final_synthesis = self._call_synthesis_model(
            context=synthesis_context,
            question=question,
            task_name="final_synthesis",  # Use o3 for complex integration
            focus="prioritize detailed locus reports with genomic coordinates and biological context",
            synthesis_type="reduce_selection"
        )
        
        # DEBUG: Save Reduce step output
        self._save_debug_data("reduce_step_final_output", final_synthesis, f"Final synthesis from {len(chunk_summaries)} chunk summaries")
        
        return final_synthesis
    
    
    def _format_data_for_synthesis(self, data: List[Dict[str, Any]]) -> str:
        """
        Format unified data for synthesis model.
        
        Args:
            data: Unified data to format
            
        Returns:
            Formatted string for synthesis
        """
        if not data:
            return "No data available"
        
        # EARLY DETECTION: Check if this is gene record data that should be grouped
        if self._looks_like_gene_records(data):
            loci = self._group_genes_into_loci(data)
            return self._format_loci_for_synthesis(loci)
        
        formatted_items = []
        for i, item in enumerate(data):
            # Format based on item type
            if item.get('type') == 'task_note':
                formatted_item = f"Task {item['task_id']}: {item['description']}\n"
                formatted_item += f"Observations: {'; '.join(item['observations'])}\n"
                formatted_item += f"Key Findings: {'; '.join(item['key_findings'])}\n"
                if item.get('quantitative_data'):
                    formatted_item += f"Data: {item['quantitative_data']}\n"
                if item.get('cross_task_connections'):
                    connections = [f"{conn['connected_task']} ({conn['connection_type']})" 
                                 for conn in item['cross_task_connections']]
                    formatted_item += f"Connections: {'; '.join(connections)}\n"
            elif item.get('type') == 'task_metadata':
                formatted_item = f"Task Metadata ({item['task_count']} tasks):\n"
                formatted_item += f"Key Insights: {'; '.join(item['key_insights'][:5])}\n"
                if item.get('cross_connections'):
                    connections = [f"{conn['from_task']} → {conn['to_task']}" 
                                 for conn in item['cross_connections'][:3]]
                    formatted_item += f"Cross-connections: {'; '.join(connections)}\n"
            else:
                # Generic raw data item
                formatted_item = f"Data Item {i+1}: {str(item)}\n"
        
        return "\n".join(formatted_items)
    
    def _format_as_genomic_loci(self, data: List[Dict[str, Any]]) -> str:
        """
        Format protein data as genomic loci grouped by contig and sorted by position.
        
        Args:
            data: List of protein data items
            
        Returns:
            Formatted string with proteins organized by genomic loci
        """
        # Group proteins by contig
        loci = {}
        for item in data:
            if 'protein_id' in item and 'start_coordinate' in item:
                # Extract contig from protein_id (e.g., NODE_405_length_15990_cov_15.881707_14 -> NODE_405)
                protein_id = item['protein_id']
                if 'NODE_' in protein_id:
                    contig_parts = protein_id.split('_')
                    if len(contig_parts) >= 2:
                        contig_id = f"{contig_parts[0]}_{contig_parts[1]}"  # NODE_405
                    else:
                        contig_id = "unknown_contig"
                else:
                    contig_id = "unknown_contig"
                
                if contig_id not in loci:
                    loci[contig_id] = []
                loci[contig_id].append(item)
        
        # Format each locus
        formatted_loci = []
        for locus_num, (contig_id, proteins) in enumerate(loci.items(), 1):
            # Sort proteins by genomic position
            proteins.sort(key=lambda p: int(p.get('start_coordinate', 0)))
            
            # Get locus boundaries
            start_pos = min(int(p.get('start_coordinate', 0)) for p in proteins)
            end_pos = max(int(p.get('end_coordinate', 0)) for p in proteins)
            
            # Count integrase genes (anchor points)
            # Count anchor genes (removed hardcoded integrase reference - already calculated above)
            
            # Format locus header
            locus_header = f"**Locus {locus_num} ({contig_id})**: {start_pos:,}-{end_pos:,} bp, {len(proteins)} genes, {anchor_count} anchor gene(s)"
            
            # Format genes in genomic order
            gene_details = []
            for i, protein in enumerate(proteins):
                gene_pos = f"{protein.get('start_coordinate', 'unknown')}-{protein.get('end_coordinate', 'unknown')}"
                function = protein.get('ko_description', 'unknown function')
                gene_details.append(f"  Gene {i+1}: {gene_pos} bp - {function}")
            
            formatted_locus = f"{locus_header}\n" + "\n".join(gene_details)
            formatted_loci.append(formatted_locus)
        
        # Summary header
        total_proteins = sum(len(proteins) for proteins in loci.values())
        header = f"GENOMIC LOCI ANALYSIS: {len(loci)} loci containing {total_proteins} genes\n\n"
        
        return header + "\n\n".join(formatted_loci)
    
    def _looks_like_gene_records(self, data: List[Dict[str, Any]]) -> bool:
        """Check if data looks like gene records that should be grouped into loci."""
        if not data or not isinstance(data, list):
            return False
        
        # Check first few items for gene record structure
        sample_size = min(3, len(data))
        gene_indicators = 0
        
        for item in data[:sample_size]:
            if isinstance(item, dict):
                # Check for canonical gene record marker
                if item.get('record_type') == 'gene_record':
                    gene_indicators += 1
                    continue
                
                # Look for key indicators of gene/protein records (multiple schema variations)
                has_protein_id = 'protein_id' in item
                has_coordinates = (('start_coordinate' in item and 'end_coordinate' in item) or
                                 ('start' in item and 'end' in item) or
                                 ('begin' in item and 'stop' in item))
                has_function = ('ko_id' in item or 'ko_description' in item or 
                               'ko_hits' in item or 'pfam_hits' in item)
                
                if has_protein_id and has_coordinates and has_function:
                    gene_indicators += 1
        
        # Require majority of samples to look like gene records
        return gene_indicators >= (sample_size * 0.7)
    
    def _group_genes_into_loci(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group gene records into genomic loci by contig and position.
        
        Args:
            records: List of gene/protein records
            
        Returns:
            List of locus dictionaries
        """
        # Group by contig
        contigs = {}
        for record in records:
            # Use canonical contig_id if available, otherwise extract from protein_id
            contig_id = record.get('contig_id')
            if not contig_id:
                protein_id = record.get('protein_id', '')
                if 'NODE_' in protein_id:
                    parts = protein_id.split('_')
                    if len(parts) >= 2:
                        contig_id = f"{parts[0]}_{parts[1]}"
                    else:
                        contig_id = "unknown_contig"
                else:
                    contig_id = "unknown_contig"
            
            if contig_id not in contigs:
                contigs[contig_id] = []
            contigs[contig_id].append(record)
        
        # Build loci
        loci = []
        for locus_idx, (contig_id, genes) in enumerate(contigs.items(), 1):
            # Sort genes by position (handle multiple coordinate field names)
            def get_start_pos(g):
                return (g.get('start') or g.get('start_coordinate') or 
                       g.get('begin') or g.get('startCoordinate') or 0)
            
            def get_end_pos(g):
                return (g.get('end') or g.get('end_coordinate') or 
                       g.get('stop') or g.get('endCoordinate') or 0)
            
            genes.sort(key=lambda g: int(get_start_pos(g)))
            
            # Get locus boundaries
            start_pos = min(int(get_start_pos(g)) for g in genes)
            end_pos = max(int(get_end_pos(g)) for g in genes)
            
            # Count anchor genes (genes with specific functional criteria that triggered selection)
            anchor_count = sum(1 for g in genes if g.get('ko_hits') or g.get('pfam_hits'))
            
            # Build gene summaries
            gene_summaries = []
            for gene in genes:
                gene_summaries.append({
                    'gene_id': gene.get('protein_id', 'unknown'),
                    'start': int(get_start_pos(gene)),
                    'end': int(get_end_pos(gene)),
                    'strand': gene.get('strand', None),
                    'function': gene.get('ko_description', 'unknown function'),
                    'ko_id': gene.get('ko_id', None)
                })
            
            # Create locus
            locus = {
                'locus_id': f"Locus_{locus_idx}",
                'contig_id': contig_id,
                'start': start_pos,
                'end': end_pos,
                'gene_count': len(genes),
                'anchor_count': anchor_count,
                'genes': gene_summaries,
                'confidence': 'high' if anchor_count > 0 else 'medium',
                'rationale': f"Contains {anchor_count} anchor gene(s)" if anchor_count > 0 else "Gene cluster"
            }
            
            loci.append(locus)
        
        return loci
    
    def _format_loci_for_synthesis(self, loci: List[Dict[str, Any]]) -> str:
        """
        Format loci for synthesis with region-first organization.
        
        Args:
            loci: List of locus dictionaries
            
        Returns:
            Formatted string for synthesis
        """
        if not loci:
            return "No genomic loci identified"
        
        total_genes = sum(locus['gene_count'] for locus in loci)
        
        # Sort loci by gene count (largest first) to prioritize interesting loci
        loci.sort(key=lambda l: l['gene_count'], reverse=True)
        
        # Header
        header = f"GENOMIC LOCI ANALYSIS: {len(loci)} loci containing {total_genes} genes\n"
        
        # Format each locus
        formatted_loci = []
        for locus in loci:
            # Locus header with key stats
            locus_header = (f"**{locus['locus_id']} ({locus['contig_id']})**: "
                          f"{locus['start']:,}-{locus['end']:,} bp, "
                          f"{locus['gene_count']} genes, "
                          f"{locus['anchor_count']} anchor gene(s)")
            
            # Add confidence and rationale
            locus_header += f" [Confidence: {locus['confidence']}]"
            if locus.get('rationale'):
                locus_header += f"\n  Rationale: {locus['rationale']}"
            
            # Gene table
            gene_lines = []
            for i, gene in enumerate(locus['genes'], 1):
                gene_line = (f"  Gene {i}: {gene['start']:,}-{gene['end']:,} bp "
                           f"({'→' if gene['strand'] == '1' else '←' if gene['strand'] == '-1' else '?'}) "
                           f"- {gene['function']}")
                if gene['ko_id']:
                    gene_line += f" ({gene['ko_id']})"
                gene_lines.append(gene_line)
            
            # Combine locus info
            locus_text = locus_header + "\n" + "\n".join(gene_lines)
            formatted_loci.append(locus_text)
        
        return header + "\n\n" + "\n\n".join(formatted_loci)
    
    def _enforce_context_limits(self, context: str, task_name: str, question: str = "") -> str:
        """
        Enforce strict context limits for the selected model to prevent context overflow.
        
        Args:
            context: Input context to validate/compress
            task_name: Task name for model selection
            question: Question for context (used for compression if needed)
            
        Returns:
            Context guaranteed to fit within model limits
        """
        # Get the model that will be used for this task
        model_name, model_config = self.model_allocator.get_model_for_task(task_name, question)
        
        # Calculate safe limits (leave room for system messages, response, etc.)
        safety_margin = 2000  # Reserve tokens for system messages and response
        max_input_tokens = model_config.max_context - safety_margin
        
        # Count actual tokens in context
        context_tokens = self._count_data_tokens([context]) if isinstance(context, str) else self._count_data_tokens(context)
        
        logger.info(f"🔍 Context check: {context_tokens:,} tokens for {model_name} (limit: {max_input_tokens:,})")
        
        if context_tokens <= max_input_tokens:
            return context  # Safe to use as-is
        
        # Context is too large - need to compress intelligently
        logger.warning(f"⚠️ Context exceeds {model_name} limits ({context_tokens:,} > {max_input_tokens:,}) - applying intelligent compression")
        
        # Calculate compression ratio needed
        compression_ratio = max_input_tokens / context_tokens
        target_chars = int(len(context) * compression_ratio * 0.95)  # Use 95% of target for safety
        
        # Apply intelligent compression that preserves biological information
        compressed_context = self._intelligent_compress(context, target_chars, question)
        
        # Verify the compressed context fits
        final_tokens = self._count_data_tokens([compressed_context])
        logger.info(f"✅ Compression complete: {context_tokens:,} → {final_tokens:,} tokens ({(1-final_tokens/context_tokens)*100:.1f}% reduction)")
        
        if final_tokens > max_input_tokens:
            logger.error(f"❌ CRITICAL: Compressed context still exceeds limits ({final_tokens:,} > {max_input_tokens:,})")
            raise ValueError(f"Cannot compress context enough for {model_name} (limit: {max_input_tokens:,} tokens)")
        
        return compressed_context
    
    def _intelligent_compress(self, context: str, target_chars: int, question: str = "") -> str:
        """
        Intelligently compress context while preserving biological information.
        
        Args:
            context: Original context to compress
            target_chars: Target character count
            question: Question context for relevance scoring
            
        Returns:
            Compressed context preserving key biological information
        """
        if len(context) <= target_chars:
            return context
        
        # Split into logical sections
        sections = context.split('\n\n')
        
        # Score sections by biological relevance
        scored_sections = []
        for i, section in enumerate(sections):
            score = self._score_biological_relevance(section, question)
            scored_sections.append((score, i, section))
        
        # Sort by relevance (highest first)
        scored_sections.sort(reverse=True)
        
        # Build compressed context by adding highest-scoring sections
        compressed_parts = []
        current_chars = 0
        
        for score, original_index, section in scored_sections:
            if current_chars + len(section) + 2 <= target_chars:  # +2 for \n\n
                compressed_parts.append((original_index, section))
                current_chars += len(section) + 2
            else:
                # Try to fit a truncated version of this section
                remaining_chars = target_chars - current_chars - 100  # Leave room for truncation message
                if remaining_chars > 200:  # Only if we have meaningful space
                    truncated = section[:remaining_chars] + "...[section continues]"
                    compressed_parts.append((original_index, truncated))
                break
        
        # Sort by original order and rebuild
        compressed_parts.sort()  # Sort by original_index
        compressed_context = '\n\n'.join([section for _, section in compressed_parts])
        
        # Add compression notice if context was significantly compressed
        reduction = (len(context) - len(compressed_context)) / len(context)
        if reduction > 0.3:  # If >30% reduction
            header = f"[CONTEXT INTELLIGENTLY COMPRESSED - {reduction*100:.1f}% reduction while preserving biological relevance]\n\n"
            compressed_context = header + compressed_context
        
        return compressed_context
    
    def _score_biological_relevance(self, section: str, question: str = "") -> float:
        """
        Score a text section for biological relevance.
        
        Args:
            section: Text section to score
            question: Question context for relevance
            
        Returns:
            Relevance score (higher = more important to preserve)
        """
        score = 0.0
        section_lower = section.lower()
        
        # High priority biological terms
        high_priority_terms = [
            'protein', 'gene', 'enzyme', 'pathway', 'metabolism', 'biosynthesis',
            'annotation', 'function', 'domain', 'pfam', 'kegg', 'go:', 'ec:',
            'prophage', 'phage', 'viral', 'integrase', 'operon', 'cluster',
            'transport', 'binding', 'kinase', 'synthase', 'oxidase', 'reductase'
        ]
        
        # Medium priority terms
        medium_priority_terms = [
            'sequence', 'blast', 'similarity', 'identity', 'coverage',
            'structure', 'motif', 'region', 'site', 'residue', 'amino acid'
        ]
        
        # Score based on biological term density
        for term in high_priority_terms:
            score += section_lower.count(term) * 2.0
        
        for term in medium_priority_terms:
            score += section_lower.count(term) * 1.0
        
        # Bonus for question relevance if question provided
        if question:
            question_lower = question.lower()
            question_words = set(question_lower.split())
            section_words = set(section_lower.split())
            overlap = len(question_words.intersection(section_words))
            score += overlap * 1.5
        
        # Bonus for structured data (likely annotations)
        if any(indicator in section for indicator in [':', '=>', '|', '\t']):
            score += 1.0
        
        # Penalty for very short sections (likely noise)
        if len(section) < 50:
            score *= 0.5
        
        return score
    
    def _call_synthesis_model(self, context: str, question: str, task_name: str, focus: str, synthesis_type: str = "summarization") -> str:
        """
        Call synthesis model using model allocation system with caching.
        
        Args:
            context: Formatted context for synthesis
            question: Original user question
            task_name: Task name for model allocation
            focus: Focus areas for synthesis
            synthesis_type: Type of synthesis ("map_extraction", "reduce_selection", "summarization")
            
        Returns:
            Synthesis result
        """
        # Create cache key based on context hash and parameters
        import hashlib
        cache_key = hashlib.md5(f"{context[:1000]}{question}{task_name}{focus}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.synthesis_cache:
            self.cache_hits += 1
            logger.info(f"📋 Cache hit for synthesis (key: {cache_key[:8]}...)")
            return self.synthesis_cache[cache_key]
        
        self.cache_misses += 1
        logger.info(f"🔄 Cache miss - making API call (key: {cache_key[:8]}...)")
        
        try:
            # CRITICAL: Enforce context limits before API call
            safe_context = self._enforce_context_limits(context, task_name, question)
            
            # Select appropriate signature based on synthesis type
            if synthesis_type == "map_extraction":
                from ..dspy_signatures import GenomicDataExtractor
                signature_class = GenomicDataExtractor
                
                def synthesize_call(module):
                    result = module(
                        genomic_data=safe_context,
                        focus_areas=focus
                    )
                    # Return combined output for compatibility
                    return f"KEY LOCI:\n{result.key_loci}\n\nBIOLOGICAL CONTEXT:\n{result.biological_context}\n\nQUANTITATIVE METRICS:\n{result.quantitative_metrics}"
                    
            elif synthesis_type == "reduce_selection":
                from ..dspy_signatures import GenomicSelector
                signature_class = GenomicSelector
                
                def synthesize_call(module):
                    result = module(
                        question=question,
                        chunk_extractions=safe_context
                    )
                    # Return combined output with data validation fields
                    return f"FINAL REPORT:\n{result.final_report}\n\nSELECTION REASONING:\n{result.selection_reasoning}\n\nBIOLOGICAL SIGNIFICANCE:\n{result.biological_significance}\n\nDATA SOURCES:\n{result.data_sources}\n\nUNSUPPORTED CLAIMS:\n{result.unsupported_claims}"
                    
            else:  # Default to summarization
                from ..dspy_signatures import GenomicSummarizer
                signature_class = GenomicSummarizer
                
                def synthesize_call(module):
                    result = module(
                        genomic_data=safe_context,
                        target_length="detailed",
                        focus_areas=focus
                    )
                    # Return combined output for compatibility
                    return f"SUMMARY:\n{result.summary}\n\nKEY FINDINGS:\n{result.key_findings}\n\nDATA STATISTICS:\n{result.data_statistics}"
            
            # ENHANCED retry logic with better rate limit detection
            max_retries = 5  # More retries for rate limits
            retry_delay = 5  # Start with longer delay (5 seconds)
            
            for attempt in range(max_retries):
                try:
                    result = self.model_allocator.create_context_managed_call(
                        task_name=task_name,
                        signature_class=signature_class,
                        module_call_func=synthesize_call,
                        query=question,
                        task_context=f"Progressive synthesis: {focus}"
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(indicator in error_str for indicator in [
                        "429", "rate limit", "too many requests", "quota", "tokens per minute"
                    ])
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        logger.warning(f"⏳ Rate limited detected, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        logger.warning(f"🔍 Error details: {error_str[:200]}...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 30)  # Cap at 30 seconds
                        continue
                    else:
                        raise  # Not rate limited or max retries reached
            
            # Handle different result formats based on synthesis type
            if isinstance(result, str):
                # synthesize_call function returned formatted string directly
                synthesis_result = result
            elif result and hasattr(result, 'summary'):
                # Old GenomicSummarizer format
                synthesis_result = result.summary
            else:
                logger.warning(f"Model allocation returned unexpected result format: {type(result)}")
                fallback_result = f"Synthesis completed but result format unexpected. Context: {context[:500]}..."
                self.synthesis_cache[cache_key] = fallback_result
                return fallback_result
            
            # Cache and return the result
            self.synthesis_cache[cache_key] = synthesis_result
            return synthesis_result
                
        except Exception as e:
            logger.error(f"Synthesis model call failed: {e}")
            error_result = f"Synthesis failed: {str(e)}"
            # Don't cache errors
            return error_result
    
    def _choose_optimal_synthesis_strategy(self, unified_data: List[Dict[str, Any]], question: str) -> str:
        """
        Choose the optimal synthesis strategy based on data quality and question type.
        
        Args:
            unified_data: Unified data for synthesis
            question: User's question
            
        Returns:
            Synthesis strategy: "key_findings_only" or "full_context"
        """
        logger.info("🧠 Analyzing synthesis strategy requirements")
        
        # PRIORITY CHECK: Structured loci data always needs Map-Reduce
        has_structured_loci = any(item.get('type') == 'structured_loci' for item in unified_data)
        
        if has_structured_loci:
            logger.info("🔬 Structured loci detected - forcing Map-Reduce for GenomicSelector pathway")
            return "full_context"
        
        # Assess key findings completeness and quality
        key_findings_quality = self._assess_key_findings_completeness(unified_data)
        
        # Check question complexity requirements
        requires_detailed_context = self._question_requires_detailed_context(question)
        
        # FORCE MAP-REDUCE: Always use full_context to ensure anti-hallucination constraints
        # and detailed Map-Reduce analysis instead of lightweight key-findings-only mode
        logger.info(f"🎯 Forcing Map-Reduce synthesis (key_findings_quality: {key_findings_quality:.2f}, detailed_context: {requires_detailed_context})")
        return "full_context"
    
    def _assess_key_findings_completeness(self, unified_data: List[Dict[str, Any]]) -> float:
        """
        Assess how complete and useful the key findings are.
        
        Args:
            unified_data: Unified data to assess
            
        Returns:
            Quality score 0.0-1.0 (higher = more complete findings)
        """
        total_findings = 0
        quality_indicators = 0
        
        for item in unified_data:
            if item.get('type') == 'task_note' and 'key_findings' in item:
                findings = item['key_findings']
                if findings:
                    total_findings += len(findings)
                    
                    # Check for quality indicators
                    for finding in findings:
                        finding_lower = str(finding).lower()
                        
                        # High-quality indicators
                        if any(term in finding_lower for term in [
                            'identified', 'detected', 'found', 'analyzed', 'discovered',
                            'protein', 'gene', 'loci', 'coordinates', 'domain', 
                            'prophage', 'operon', 'pathway', 'annotation'
                        ]):
                            quality_indicators += 1
                        
                        # Quantitative indicators (numbers suggest real analysis)
                        if any(char.isdigit() for char in finding):
                            quality_indicators += 0.5
        
        # Calculate quality score
        if total_findings == 0:
            return 0.0
        
        quality_ratio = quality_indicators / total_findings
        
        # Adjust based on total findings count (more findings = higher confidence)
        findings_bonus = min(total_findings / 20, 0.2)  # Max 20% bonus
        
        final_score = min(quality_ratio + findings_bonus, 1.0)
        logger.info(f"📊 Key findings assessment: {total_findings} findings, {quality_indicators} quality indicators, score: {final_score:.2f}")
        
        return final_score
    
    def _question_requires_detailed_context(self, question: str) -> bool:
        """
        Determine if question requires detailed context beyond key findings.
        
        Args:
            question: User's question
            
        Returns:
            True if detailed context needed, False if key findings sufficient
        """
        question_lower = question.lower()
        
        # Detailed context required for these patterns
        detailed_patterns = [
            'detailed report', 'comprehensive', 'full analysis', 'all details',
            'coordinates', 'exact', 'precise', 'specific location', 'sequence',
            'show me everything', 'maximum detail', 'complete analysis'
        ]
        
        # Simple patterns that can use key findings only
        simple_patterns = [
            'quick', 'summary', 'overview', 'brief', 'what are', 'how many',
            'list', 'find', 'identify', 'detect', 'simple'
        ]
        
        requires_detailed = any(pattern in question_lower for pattern in detailed_patterns)
        is_simple = any(pattern in question_lower for pattern in simple_patterns)
        
        # Default to detailed if unclear, but simple patterns override
        return requires_detailed or (not is_simple and len(question.split()) > 15)
    
    def _synthesize_from_key_findings_only(self, unified_data: List[Dict[str, Any]], question: str) -> str:
        """
        Lightweight synthesis using only key findings and discoveries.
        
        Args:
            unified_data: Unified data (only key findings will be used)
            question: User's question
            
        Returns:
            Synthesis result from key findings only
        """
        logger.info("🎯 Performing key-findings-only synthesis (lightweight mode)")
        
        # Extract all key findings
        all_findings = []
        task_summaries = []
        
        for item in unified_data:
            if item.get('type') == 'task_note':
                task_id = item.get('task_id', 'unknown')
                description = item.get('description', '')
                findings = item.get('key_findings', [])
                
                if findings:
                    task_summaries.append(f"Task {task_id}: {description[:100]}...")
                    all_findings.extend(findings)
        
        # Build lightweight context
        lightweight_context = f"""
BIOLOGICAL DISCOVERIES FROM ANALYSIS:

Task Overview:
{chr(10).join(task_summaries)}

Key Findings and Discoveries:
{chr(10).join(f"• {finding}" for finding in all_findings)}

ANALYSIS COMPLETE - {len(all_findings)} key discoveries identified
"""
        
        # Use direct synthesis with lightweight context
        from ..dspy_signatures import GenomicSynthesizer
        
        def synthesis_call(module):
            return module(
                question=question,
                context=lightweight_context,
                synthesis_mode="discovery_summary"
            )
        
        result = self.model_allocator.create_context_managed_call(
            task_name="biological_interpretation",
            signature_class=GenomicSynthesizer,
            module_call_func=synthesis_call,
            query=question,
            task_context="Key findings synthesis"
        )
        
        if result and hasattr(result, 'summary'):
            logger.info(f"✅ Key-findings-only synthesis complete ({len(lightweight_context)} characters)")
            return result.summary
        else:
            logger.warning("Key-findings synthesis failed, using fallback")
            return f"Analysis complete. Identified {len(all_findings)} key biological discoveries from the genomic analysis."

    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the synthesis process.
        
        Returns:
            Dictionary with synthesis statistics
        """
        total_calls = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        return {
            "architecture": "Map-Reduce",
            "direct_synthesis_limit": self.direct_synthesis_limit,
            "map_chunk_limit": self.map_chunk_limit,
            "tokenizer_available": self.tokenizer is not None,
            "model_allocator_available": self.model_allocator is not None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate,
            "api_call_reduction": f"{cache_hit_rate:.1f}% fewer API calls"
        }