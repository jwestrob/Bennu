"""
Multi-part report synthesizer for handling large-scale genomic analyses.

Extends the progressive synthesizer to handle reports that exceed token limits
by creating structured, navigable multi-part reports with executive summaries
and comprehensive synthesis sections.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tiktoken

from .progressive_synthesizer import ProgressiveSynthesizer
from .note_keeper import NoteKeeper
from .note_schemas import TaskNote, SynthesisNote, ConfidenceLevel
from .report_manager import ReportPlanner, ReportPlan, ReportChunk

logger = logging.getLogger(__name__)


class MultiPartReportSynthesizer(ProgressiveSynthesizer):
    """
    Extends ProgressiveSynthesizer to handle multi-part reports.
    
    Manages the creation of comprehensive reports that exceed token limits
    by intelligently chunking data and creating structured, navigable reports
    with executive summaries and synthesis sections.
    """
    
    def __init__(self, note_keeper: NoteKeeper, chunk_size: int = 8, 
                 max_part_tokens: int = 18000):
        """
        Initialize multi-part report synthesizer.
        
        Args:
            note_keeper: NoteKeeper instance for accessing notes
            chunk_size: Number of tasks to process per chunk
            max_part_tokens: Maximum tokens per report part
        """
        super().__init__(note_keeper, chunk_size)
        self.max_part_tokens = max_part_tokens
        self.report_planner = ReportPlanner(max_part_tokens)
        
        # Initialize DSPy modules for multi-part reports (will use global configuration)
        self.report_part_generator = None
        self.executive_summary_generator = None
        self.report_synthesis_generator = None
        
        # Initialize basic synthesizer for fallback
        self.synthesizer = None
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def initialize_dspy_modules(self, dspy_module):
        """Initialize DSPy modules for multi-part report generation using global configuration."""
        try:
            import dspy
            from ..dspy_signatures import (
                ReportPartGenerator, 
                ExecutiveSummaryGenerator, 
                ReportSynthesisGenerator
            )
            
            # Use global DSPy configuration (same as main system)
            self.report_part_generator = dspy.Predict(ReportPartGenerator)
            self.executive_summary_generator = dspy.Predict(ExecutiveSummaryGenerator)
            self.report_synthesis_generator = dspy.Predict(ReportSynthesisGenerator)
            
            # Initialize basic synthesizer for fallback to standard synthesis
            from ..dspy_signatures import GenomicSummarizer
            self.synthesizer = dspy.Predict(GenomicSummarizer)
            
            logger.info("📄 Multi-part report DSPy modules initialized with global configuration")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-part report DSPy modules: {e}")
            
            # Set modules to None so fallback methods are used
            self.report_part_generator = None
            self.executive_summary_generator = None
            self.report_synthesis_generator = None
    
    def synthesize_multipart_report(self, 
                                   task_notes: List[TaskNote],
                                   question: str,
                                   data: List[Dict[str, Any]]) -> str:
        """
        Generate a multi-part report from task notes and data.
        
        Args:
            task_notes: List of TaskNote objects
            question: Original user question
            data: Raw data for the report
            
        Returns:
            Formatted multi-part report
        """
        logger.info(f"📄 Starting multi-part report generation for {len(data)} data points")
        
        # Plan the report structure
        report_plan = self.report_planner.plan_report(question, data, task_notes)
        
        if not report_plan.requires_chunking:
            logger.info("📄 Report size manageable, using single-part generation")
            # Fall back to standard progressive synthesis without DSPy parameter
            return self._synthesize_standard_fallback(task_notes, question)
        
        logger.info(f"📄 Report requires chunking: {len(report_plan.chunks)} parts planned")
        
        # Generate multi-part report
        return self._generate_multipart_report(question, report_plan, task_notes, data)
    
    def _generate_multipart_report(self, 
                                  question: str,
                                  report_plan: ReportPlan,
                                  task_notes: List[TaskNote],
                                  data: List[Dict[str, Any]]) -> str:
        """Generate the complete multi-part report."""
        report_parts = []
        
        # 1. Generate executive summary
        executive_summary = self._generate_executive_summary(question, report_plan, data)
        report_parts.append(("📄 Executive Summary", executive_summary))
        
        # 2. Generate each report part
        previous_parts_summary = ""
        for i, chunk in enumerate(report_plan.chunks):
            logger.info(f"📄 Generating part {i+1}/{len(report_plan.chunks)}: {chunk.title}")
            
            part_content = self._generate_report_part(
                question, chunk, previous_parts_summary, report_plan.report_type.value
            )
            
            part_title = f"📖 Part {i+1}: {chunk.title}"
            report_parts.append((part_title, part_content))
            
            # Update previous parts summary
            previous_parts_summary = self._update_previous_parts_summary(
                previous_parts_summary, part_content
            )
        
        # 3. Generate synthesis section
        synthesis_section = self._generate_synthesis_section(
            question, report_parts, task_notes, report_plan
        )
        report_parts.append(("🔬 Synthesis & Conclusions", synthesis_section))
        
        # 4. Format the complete report
        return self._format_multipart_report(report_parts, report_plan)
    
    def _generate_executive_summary(self, 
                                   question: str,
                                   report_plan: ReportPlan,
                                   data: List[Dict[str, Any]]) -> str:
        """Generate executive summary for the report."""
        if not self.executive_summary_generator:
            return self._generate_fallback_executive_summary(question, report_plan, data)
        
        try:
            # Prepare data overview
            data_overview = self._prepare_data_overview(data)
            
            # Identify key patterns
            key_patterns = self._identify_key_patterns(data)
            
            # Create report structure description
            report_structure = self._describe_report_structure(report_plan)
            
            # Generate executive summary
            result = self.executive_summary_generator(
                question=question,
                data_overview=data_overview,
                key_patterns=key_patterns,
                report_structure=report_structure
            )
            
            return self._format_executive_summary(result)
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return self._generate_fallback_executive_summary(question, report_plan, data)
    
    def _generate_report_part(self, 
                             question: str,
                             chunk: ReportChunk,
                             previous_parts_summary: str,
                             report_type: str) -> str:
        """Generate content for a specific report part."""
        if not self.report_part_generator:
            return self._generate_fallback_report_part(chunk)
        
        try:
            # Format data chunk for processing
            data_chunk_formatted = self._format_data_chunk(chunk.data_subset)
            
            # Generate part content
            result = self.report_part_generator(
                question=question,
                data_chunk=data_chunk_formatted,
                part_context=chunk.context,
                previous_parts_summary=previous_parts_summary,
                report_type=report_type
            )
            
            return self._format_report_part(result, chunk)
            
        except Exception as e:
            logger.error(f"Failed to generate report part {chunk.chunk_id}: {e}")
            return self._generate_fallback_report_part(chunk)
    
    def _generate_synthesis_section(self, 
                                   question: str,
                                   report_parts: List[Tuple[str, str]],
                                   task_notes: List[TaskNote],
                                   report_plan: ReportPlan) -> str:
        """Generate final synthesis section."""
        if not self.report_synthesis_generator:
            return self._generate_fallback_synthesis(report_parts)
        
        try:
            # Summarize all parts
            all_parts_summary = self._summarize_all_parts(report_parts)
            
            # Identify cross-cutting themes
            cross_cutting_themes = self._identify_cross_cutting_themes(report_parts, task_notes)
            
            # Create quantitative integration
            quantitative_integration = self._create_quantitative_integration(report_parts)
            
            # Generate synthesis
            result = self.report_synthesis_generator(
                question=question,
                all_parts_summary=all_parts_summary,
                cross_cutting_themes=cross_cutting_themes,
                quantitative_integration=quantitative_integration
            )
            
            return self._format_synthesis_section(result)
            
        except Exception as e:
            logger.error(f"Failed to generate synthesis section: {e}")
            return self._generate_fallback_synthesis(report_parts)
    
    def _synthesize_standard_fallback(self, 
                                     task_notes: List[TaskNote],
                                     question: str) -> str:
        """Fallback synthesis without passing DSPy synthesizer parameter."""
        if self.synthesizer:
            # Use the initialized synthesizer
            from .progressive_synthesizer import ProgressiveSynthesizer
            temp_synthesizer = ProgressiveSynthesizer(self.note_keeper, self.chunk_size)
            return temp_synthesizer._synthesize_standard(task_notes, self.synthesizer, question)
        else:
            # Basic fallback without DSPy
            return f"Analysis of {len(task_notes)} tasks completed. DSPy unavailable for detailed synthesis."
    
    def _prepare_data_overview(self, data: List[Dict[str, Any]]) -> str:
        """Prepare statistical overview of the dataset."""
        overview_parts = []
        
        # Basic statistics
        overview_parts.append(f"Dataset size: {len(data)} data points")
        
        # Genome count
        genomes = set()
        for item in data:
            genome_id = item.get('genome_id', item.get('genome', 'unknown'))
            genomes.add(genome_id)
        overview_parts.append(f"Genomes analyzed: {len(genomes)}")
        
        # Data types
        data_types = set()
        for item in data:
            if 'ko_id' in item:
                data_types.add('KEGG Orthology')
            if 'cazyme_family' in item:
                data_types.add('CAZyme families')
            if 'transport_type' in item:
                data_types.add('Transport systems')
        
        if data_types:
            overview_parts.append(f"Data types: {', '.join(data_types)}")
        
        return " | ".join(overview_parts)
    
    def _identify_key_patterns(self, data: List[Dict[str, Any]]) -> str:
        """Identify key patterns in the dataset."""
        patterns = []
        
        # Distribution patterns
        genome_counts = {}
        for item in data:
            genome_id = item.get('genome_id', 'unknown')
            genome_counts[genome_id] = genome_counts.get(genome_id, 0) + 1
        
        if genome_counts:
            sorted_genomes = sorted(genome_counts.items(), key=lambda x: x[1], reverse=True)
            patterns.append(f"Most data from {sorted_genomes[0][0]} ({sorted_genomes[0][1]} entries)")
        
        # Functional patterns
        functions = {}
        for item in data:
            func = item.get('ko_description', item.get('function', 'unknown'))
            functions[func] = functions.get(func, 0) + 1
        
        if functions:
            top_functions = sorted(functions.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns.append(f"Top functions: {', '.join([f[0] for f in top_functions])}")
        
        return " | ".join(patterns) if patterns else "No clear patterns identified"
    
    def _describe_report_structure(self, report_plan: ReportPlan) -> str:
        """Describe the structure of the multi-part report."""
        structure_parts = []
        
        structure_parts.append(f"Report type: {report_plan.report_type.value}")
        structure_parts.append(f"Total parts: {len(report_plan.chunks)}")
        
        # Describe chunking strategy
        if report_plan.chunks:
            chunk_type = report_plan.chunks[0].chunk_type.value
            structure_parts.append(f"Organized by: {chunk_type}")
        
        structure_parts.append(f"Estimated total tokens: {report_plan.total_estimated_tokens}")
        
        return " | ".join(structure_parts)
    
    def _format_data_chunk(self, data_chunk: List[Dict[str, Any]]) -> str:
        """Format data chunk for report generation."""
        if not data_chunk:
            return "No data available for this section"
        
        # Create a more digestible summary instead of raw data dump
        formatted_parts = []
        formatted_parts.append(f"Data subset: {len(data_chunk)} items")
        
        # Summarize key patterns instead of showing raw data
        if data_chunk:
            # Count genomes
            genomes = set()
            functions = set()
            for item in data_chunk:
                if 'genome_id' in item:
                    genomes.add(item['genome_id'])
                if 'ko_description' in item:
                    functions.add(item['ko_description'])
                elif 'cazyme_family' in item:
                    functions.add(item['cazyme_family'])
            
            if genomes:
                formatted_parts.append(f"Genomes: {len(genomes)}")
            if functions:
                formatted_parts.append(f"Functions: {len(functions)}")
                # Show a few example functions instead of raw data
                example_functions = list(functions)[:3]
                formatted_parts.append(f"Examples: {', '.join(example_functions)}")
        
        return " | ".join(formatted_parts)
    
    def _format_executive_summary(self, result) -> str:
        """Format executive summary result."""
        sections = []
        
        sections.append(f"**Executive Summary**")
        sections.append(result.executive_summary)
        sections.append("")
        
        sections.append(f"**Scope & Methodology**")
        sections.append(result.scope_and_methodology)
        sections.append("")
        
        sections.append(f"**Key Statistics**")
        sections.append(result.key_statistics)
        sections.append("")
        
        sections.append(f"**Navigation Guide**")
        sections.append(result.navigation_guide)
        
        return "\n".join(sections)
    
    def _format_report_part(self, result, chunk: ReportChunk) -> str:
        """Format individual report part."""
        sections = []
        
        sections.append(f"**{chunk.title}**")
        sections.append(f"*{chunk.context}*")
        sections.append("")
        
        sections.append(result.part_content)
        sections.append("")
        
        if result.key_findings:
            sections.append("**Key Findings:**")
            sections.append(result.key_findings)
            sections.append("")
        
        if result.quantitative_summary:
            sections.append("**Quantitative Summary:**")
            sections.append(result.quantitative_summary)
            sections.append("")
        
        return "\n".join(sections)
    
    def _format_synthesis_section(self, result) -> str:
        """Format synthesis section."""
        sections = []
        
        sections.append("**Comprehensive Synthesis**")
        sections.append(result.synthesis_content)
        sections.append("")
        
        sections.append("**Biological Implications**")
        sections.append(result.biological_implications)
        sections.append("")
        
        sections.append("**Recommendations**")
        sections.append(result.recommendations)
        sections.append("")
        
        sections.append("**Confidence Assessment**")
        sections.append(result.confidence_assessment)
        
        return "\n".join(sections)
    
    def _format_multipart_report(self, 
                                report_parts: List[Tuple[str, str]],
                                report_plan: ReportPlan) -> str:
        """Format the complete multi-part report."""
        sections = []
        
        # Header
        sections.append("# 📄 Multi-Part Genomic Analysis Report")
        sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        sections.append(f"*Report Type: {report_plan.report_type.value}*")
        sections.append(f"*Total Parts: {len(report_parts)}*")
        sections.append("")
        
        # Table of contents
        sections.append("## 📋 Table of Contents")
        for i, (title, _) in enumerate(report_parts):
            sections.append(f"{i+1}. {title}")
        sections.append("")
        
        # Report parts
        for i, (title, content) in enumerate(report_parts):
            sections.append(f"## {title}")
            sections.append(content)
            sections.append("")
            
            # Add separator between parts
            if i < len(report_parts) - 1:
                sections.append("---")
                sections.append("")
        
        # Footer
        sections.append("---")
        sections.append("*End of Multi-Part Report*")
        
        return "\n".join(sections)
    
    def _update_previous_parts_summary(self, 
                                      previous_summary: str,
                                      new_part_content: str) -> str:
        """Update summary of previous parts for consistency."""
        # Extract key points from new part
        key_points = []
        lines = new_part_content.split('\n')
        for line in lines:
            if line.startswith('**Key Findings:**') or line.startswith('- '):
                key_points.append(line.strip())
        
        # Combine with previous summary
        if previous_summary:
            combined = f"{previous_summary} | Recent findings: {'; '.join(key_points[:3])}"
        else:
            combined = f"Initial findings: {'; '.join(key_points[:3])}"
        
        # Keep summary manageable
        if len(combined) > 500:
            combined = combined[:500] + "..."
        
        return combined
    
    def _summarize_all_parts(self, report_parts: List[Tuple[str, str]]) -> str:
        """Summarize all report parts for synthesis."""
        summaries = []
        
        for title, content in report_parts[1:-1]:  # Skip executive summary and synthesis
            # Extract key findings
            key_findings = []
            lines = content.split('\n')
            for line in lines:
                if '**Key Findings:**' in line:
                    key_findings.append(line.strip())
                elif line.startswith('- ') and len(key_findings) < 3:
                    key_findings.append(line.strip())
            
            if key_findings:
                summaries.append(f"{title}: {'; '.join(key_findings)}")
        
        return " | ".join(summaries)
    
    def _identify_cross_cutting_themes(self, 
                                      report_parts: List[Tuple[str, str]],
                                      task_notes: List[TaskNote]) -> str:
        """Identify themes that appear across multiple parts."""
        themes = []
        
        # Look for common terms across parts
        all_content = " ".join([content for _, content in report_parts])
        
        # Common biological themes
        theme_terms = {
            'evolution': ['evolution', 'evolutionary', 'conserved', 'phylogen'],
            'metabolism': ['metabolic', 'pathway', 'enzyme', 'biosynthesis'],
            'transport': ['transport', 'transporter', 'permease', 'channel'],
            'regulation': ['regulation', 'regulatory', 'control', 'expression'],
            'adaptation': ['adaptation', 'environmental', 'stress', 'response']
        }
        
        for theme, terms in theme_terms.items():
            if any(term in all_content.lower() for term in terms):
                themes.append(theme)
        
        return ', '.join(themes) if themes else "No clear cross-cutting themes identified"
    
    def _create_quantitative_integration(self, report_parts: List[Tuple[str, str]]) -> str:
        """Create integrated quantitative analysis."""
        quantitative_parts = []
        
        for title, content in report_parts:
            # Extract quantitative information
            lines = content.split('\n')
            for line in lines:
                if any(char.isdigit() for char in line) and any(term in line.lower() for term in ['count', 'total', 'average', 'percent']):
                    quantitative_parts.append(line.strip())
        
        return " | ".join(quantitative_parts[:5]) if quantitative_parts else "Limited quantitative data available"
    
    def _generate_fallback_executive_summary(self, 
                                           question: str,
                                           report_plan: ReportPlan,
                                           data: List[Dict[str, Any]]) -> str:
        """Generate fallback executive summary when DSPy unavailable."""
        return f"""
**Executive Summary**

This multi-part report addresses: {question}

**Scope:** Analysis of {len(data)} data points across {len(report_plan.chunks)} report sections
**Methodology:** {report_plan.report_type.value} analysis with intelligent chunking
**Structure:** {len(report_plan.chunks)} detailed parts plus synthesis

**Key Statistics:**
- Total data points: {len(data)}
- Estimated analysis scope: {report_plan.total_estimated_tokens} tokens
- Report organization: {report_plan.chunks[0].chunk_type.value if report_plan.chunks else 'comprehensive'}

**Navigation Guide:**
Each part provides detailed analysis of a specific subset of the data, maintaining scientific rigor while managing computational constraints.
"""
    
    def _generate_fallback_report_part(self, chunk: ReportChunk) -> str:
        """Generate fallback report part when DSPy unavailable."""
        # Create a simple analysis without DSPy
        data_summary = self._format_data_chunk(chunk.data_subset)
        
        return f"""
**{chunk.title}**

*{chunk.context}*

This section analyzes {len(chunk.data_subset)} data points using {chunk.chunk_type.value} organization.

**Data Overview:**
{data_summary}

**Analysis:** Basic analysis without advanced language model

**Key Findings:** This section contains functional annotations and protein data that would benefit from detailed analysis with advanced language models.
"""
    
    def _generate_fallback_synthesis(self, report_parts: List[Tuple[str, str]]) -> str:
        """Generate fallback synthesis when DSPy unavailable."""
        return f"""
**Comprehensive Synthesis**

This report analyzed data across {len(report_parts) - 2} detailed sections.

**Integration:** DSPy unavailable - comprehensive synthesis not generated

**Biological Implications:** Analysis requires DSPy for biological interpretation

**Recommendations:** Enable DSPy for detailed recommendations

**Confidence Assessment:** Low - fallback synthesis mode
"""