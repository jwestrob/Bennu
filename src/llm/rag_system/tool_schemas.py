#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GeneContextModel(BaseModel):
    gene_id: str
    protein_id: Optional[str] = None
    start: int
    end: int
    strand: str
    length: int
    annotation: Optional[str] = None
    ko_id: Optional[str] = None
    ko_description: Optional[str] = None
    pfam_domains: List[str] = Field(default_factory=list)
    is_hypothetical: Optional[bool] = None


class ContigContextModel(BaseModel):
    contig_id: str
    length: Optional[int] = None
    plus_strand_genes: List[GeneContextModel] = Field(default_factory=list)
    minus_strand_genes: List[GeneContextModel] = Field(default_factory=list)
    total_genes: int


class GenomeContextModel(BaseModel):
    genome_id: str
    contigs: List[ContigContextModel] = Field(default_factory=list)
    total_genes: int
    total_contigs: int
    annotated_gene_count: Optional[int] = None
    largest_contig_length: Optional[int] = None


class LiteratureArticleModel(BaseModel):
    pmid: Optional[str] = None
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None


class DatabaseQueryResultModel(BaseModel):
    cypher: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result_count: Optional[int] = None
    rows: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None


class CodeInterpreterResultModel(BaseModel):
    session_id: Optional[str] = None
    success: bool = False
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    artifacts: Optional[List[Dict[str, Any]]] = None
    plots: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


class GenomeSelectorResultModel(BaseModel):
    success: bool = False
    selected_genome: Optional[str] = None
    match_score: Optional[float] = None
    match_reason: Optional[str] = None
    available_genomes: Optional[List[str]] = None
    error_message: Optional[str] = None


class Claim(BaseModel):
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    risk_level: Optional[str] = None


class SynthesisInput(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    disclaimers: Optional[List[str]] = None
    notes_ref: Optional[str] = None


class ToolResultEnvelope(BaseModel):
    tool_name: str
    success: bool
    version: str = "1.0"
    tool_result_id: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    display_text: Optional[str] = None
    structured_data: Optional[List[Any]] = None
    references: List[str] = Field(default_factory=list)
    timings: Optional[Dict[str, float]] = None
    token_usage: Optional[Dict[str, int]] = None

