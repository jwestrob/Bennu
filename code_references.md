# Code Reference Analysis

### annotation_processors.py
- [x] `AnnotationProcessor`
- [x] `PfamProcessor`
- [x] `KofamProcessor`
- [x] `process_astra_results()`
- [ ] `main()`

### batch_processor.py
- [x] `get_neo4j_driver()`
- [x] `BatchQuery`
- [x] `BatchResult`
- [x] `BatchJob`
- [x] `BatchQueryProcessor`

### functional_enrichment.py
- [x] `PfamEntry`
- [x] `KoEntry`
- [x] `CazyEntry`
- [x] `FunctionalEnrichment`
- [x] `add_functional_enrichment_to_pipeline()`

### neo4j_bulk_loader.py
- [x] `Neo4jBulkLoader`
- [ ] `main()`

### pathway_integration.py
- [x] `PathwayInfo`
- [x] `KEGGPathwayIntegrator`
- [x] `integrate_pathways()`

### provenance.py
- [ ] `ActivityEntity`
- [ ] `ProvenanceTracker`
- [ ] `create_pipeline_provenance()`
- [ ] `add_software_agent()`

### quast_parser.py
- [x] `parse_quast_report()`
- [x] `extract_genome_id_from_quast_path()`
- [x] `collect_all_quast_metrics()`
- [ ] `validate_quality_metrics()`
- [x] `format_quality_metrics_for_rdf()`

### rdf_builder.py
- [x] `parse_prodigal_header()`
- [x] `build_protein_to_genome_mapping()`
- [x] `build_contig_to_genome_index_from_proteins()`
- [x] `assign_bgc_to_correct_genome()`
- [x] `GenomeKGBuilder`
- [x] `build_knowledge_graph_from_pipeline()`
- [x] `build_knowledge_graph_with_extended_annotations()`
- [ ] `build_knowledge_graph_with_bgc()`

### rdf_to_csv_converter.py
- [x] `RDFToCSVConverter`
- [ ] `main()`

### schema.py
- [ ] `BiologicalSequenceType`
- [ ] `TaxonomicRank`
- [ ] `GenomeEntity`
- [ ] `GeneEntity`
- [ ] `ProteinEntity`
- [ ] `TaxonomicEntity`

### sequence_db_builder.py
- [ ] `SequenceDatabaseBuilder`
- [ ] `main()`

### sequence_db.py
- [ ] `SequenceDatabase`
- [ ] `get_default_sequence_db()`

### __init__.py

### client.py
- [ ] `CodeInterpreterClient`
- [ ] `GenomicCodeInterpreter`

### sequence_service.py
- [ ] `get_default_sequence_db()`
- [ ] `SequenceInfo`
- [ ] `SequenceService`
- [ ] `SequenceAnalyzer`
- [ ] `get_sequence_service()`

### service.py
- [ ] `CodeExecutionRequest`
- [ ] `CodeExecutionResponse`
- [ ] `SessionManager`
- [ ] `SecureCodeExecutor`

### 00_prepare_inputs.py
- [ ] `validate_fasta_format()`
- [ ] `calculate_file_checksum()`
- [ ] `find_genome_files()`
- [ ] `generate_genome_id()`
- [ ] `prepare_inputs()`
- [ ] `main()`

### 01_run_quast.py
- [ ] `parse_quast_report()`
- [ ] `validate_quast_outputs()`
- [ ] `run_quast_single()`
- [ ] `process_genomes_parallel()`
- [ ] `generate_summary_stats()`
- [ ] `run_quast()`
- [ ] `main()`

### 02_dfast_qc.py
- [ ] `parse_dqc_json()`
- [ ] `run_dfast()`
- [ ] `run_dfast_single()`
- [ ] `process_genomes_parallel()`
- [ ] `generate_summary_stats()`
- [ ] `call()`
- [ ] `main()`

### 03_prodigal.py
- [ ] `parse_prodigal_stats()`
- [ ] `count_sequences_in_fasta()`
- [ ] `validate_prodigal_outputs()`
- [ ] `run_prodigal_single()`
- [ ] `process_genomes_parallel()`
- [ ] `create_protein_symlinks()`
- [ ] `generate_summary_stats()`
- [ ] `run_prodigal()`
- [ ] `main()`

### 04_astra_scan.py
- [ ] `run_single_astra_scan()`
- [ ] `run_astra_scan()`
- [ ] `main()`

### 06_esm2_embeddings.py
- [ ] `ProteinSequence`
- [ ] `ESM2EmbeddingGenerator`
- [ ] `load_protein_sequences()`
- [ ] `load_fasta_sequences()`
- [ ] `create_lancedb_table()`
- [ ] `save_embeddings_and_db()`
- [ ] `run_esm2_embeddings()`

### dbcan_cazyme.py
- [ ] `CAZymeAnnotation`
- [ ] `CAZymeResult`
- [ ] `run_dbcan_analysis()`
- [ ] `parse_dbcan_overview()`
- [ ] `load_cazyme_substrate_mapping()`
- [ ] `get_cazyme_family_type()`
- [ ] `count_proteins_in_fasta()`
- [ ] `run_single_dbcan_analysis()`
- [ ] `run_dbcan_batch_analysis()`
- [ ] `save_results()`
- [ ] `create_processing_manifest()`
- [ ] `main()`

### gecco_bgc.py
- [ ] `run_gecco()`
- [ ] `create_empty_gecco_output()`
- [ ] `parse_gecco_clusters_tsv()`
- [ ] `convert_gecco_to_genbank()`
- [ ] `parse_gecco_genbank()`
- [ ] `process_genome_gecco()`
- [ ] `gecco_bgc_detection()`
- [ ] `main()`

### __init__.py

### annotation_tools.py
- [ ] `OperonAssessment`

### cli.py
- [ ] `setup_logging()`
- [ ] `ask()`
- [ ] `interactive()`
- [ ] `demo()`
- [ ] `health()`
- [ ] `config()`
- [ ] `_display_answer()`
- [ ] `_show_help()`
- [ ] `_show_examples()`

### config.py
- [ ] `DatabaseConfig`
- [ ] `LLMConfig`

### context_compression.py
- [ ] `CompressionLevel`
- [ ] `CompressionResult`
- [ ] `TokenCounter`
- [ ] `GenomicDataCompressor`
- [ ] `ContextCompressor`

### domain_functions.py
- [ ] `extract_domains_from_ids()`
- [ ] `annotate_protein_domains()`
- [ ] `format_domain_annotation()`

### error_patterns.py
- [ ] `ErrorPatternRegistry`
- [ ] `RelationshipMapper`
- [ ] `EntitySuggester`

### model_switcher.py
- [ ] `get_config()`
- [ ] `reconfigure_dspy()`
- [ ] `switch_to_cost_effective()`
- [ ] `switch_to_premium()`
- [ ] `get_current_model_status()`
- [ ] `print_model_status()`
- [ ] `configure_models()`

### pathway_tools.py
- [ ] `KEGGPathwayMapper`

### qa_chain.py
- [ ] `GenomicQAChain`
- [ ] `create_qa_chain()`
- [ ] `ComposeAnswer`

### query_processor.py
- [ ] `QueryResult`
- [ ] `BaseQueryProcessor`
- [ ] `Neo4jQueryProcessor`
- [ ] `LanceDBQueryProcessor`
- [ ] `HybridQueryProcessor`

### rag_system.py

### repair_types.py
- [ ] `RepairStrategy`
- [ ] `RepairResult`
- [ ] `ErrorPattern`
- [ ] `SchemaInfo`

### retrieval.py
- [ ] `FAISSRetriever`
- [ ] `Neo4jRetriever`
- [ ] `HybridRetriever`

### sequence_tools.py
- [ ] `_safe_log_data()`
- [ ] `extract_organism_from_id()`
- [ ] `extract_protein_ids_from_analysis()`

### task_notes.py
- [ ] `TaskNotesManager`

### task_repair_agent.py
- [ ] `TaskRepairAgent`

### __init__.py

### agent_tool_selector.py
- [ ] `ToolSelectionResult`
- [ ] `BiologicalToolSelector`
- [ ] `IntelligentToolSelector`
- [ ] `CachedToolSelector`
- [ ] `get_tool_selector()`
- [ ] `get_cached_tool_selector()`

### code_enhancement.py
- [ ] `CodeEnhancer`
- [ ] `get_distribution_by_genome()`
- [ ] `get_functional_summary()`
- [ ] `analyze_data()`
- [ ] `SequenceDatabaseConnector`
- [ ] `get_protein_sequence()`
- [ ] `analyze_amino_acid_composition()`
- [ ] `extract_protein_ids_from_task_results()`

### context_compression.py
- [ ] `CompressionStats`
- [ ] `ContextCompressor`

### context_processing.py
- [ ] `ContextProcessor`
- [ ] `ContextFormatter`

### core.py
- [ ] `GenomicRAG`

### data_scaling.py
- [ ] `DataScalingStrategy`
- [ ] `SmallDatasetStrategy`
- [ ] `analyze_protein_composition()`
- [ ] `create_protein_summary()`
- [ ] `MediumDatasetStrategy`
- [ ] `analyze_sample_batch()`
- [ ] `create_scaling_summary()`
- [ ] `LargeDatasetStrategy`
- [ ] `ScalingRouter`
- [ ] `convert_to_count_query()`
- [ ] `convert_to_aggregated_query()`

### dspy_signatures.py
- [ ] `PlannerAgent`
- [ ] `QueryClassifier`
- [ ] `ContextRetriever`
- [ ] `RelevanceValidator`
- [ ] `GenomicAnswerer`
- [ ] `GenomicSummarizer`
- [ ] `NotingDecision`
- [ ] `ReportPartGenerator`
- [ ] `ExecutiveSummaryGenerator`
- [ ] `ReportSynthesisGenerator`
- [ ] `GenomicQuery`
- [ ] `TaxonomicClassification`
- [ ] `FunctionalAnnotation`
- [ ] `ComparativeGenomics`
- [ ] `MetabolicPathway`
- [ ] `GenomeQuality`
- [ ] `GenomeSelectionSignature`

### external_tools.py
- [ ] `literature_search()`
- [ ] `report_synthesis_tool()`
- [ ] `register_tool()`
- [ ] `get_tool()`
- [ ] `list_available_tools()`

### genome_context_extractor.py
- [ ] `GenomeContext`
- [ ] `GenomeContextExtractor`

### genome_selection.py
- [ ] `GenomeSelectionResult`
- [ ] `GenomeScope`
- [ ] `UnifiedGenomeSelector`
- [ ] `get_genome_selector()`
- [ ] `set_genome_selector()`

### intelligent_chunking_manager.py
- [ ] `AnalysisChunk`
- [ ] `ChunkingStrategy`
- [ ] `IntelligentChunkingManager`

### intelligent_routing.py
- [ ] `QueryComplexity`
- [ ] `QueryScope`
- [ ] `QueryAnalysis`
- [ ] `IntelligentRouter`

### intelligent_task_splitter.py
- [ ] `SplitTaskResult`
- [ ] `IntelligentTaskSplitter`

### log_formatter.py
- [ ] `PipelineLogFormatter`
- [ ] `TaskGraphLogFilter`
- [ ] `setup_enhanced_logging()`
- [ ] `enable_clean_logging()`
- [ ] `export_task_summary()`

### policy_engine.py
- [ ] `PolicyConfig`
- [ ] `PolicyEngine`
- [ ] `get_policy_engine()`
- [ ] `load_user_config()`
- [ ] `save_user_config()`
- [ ] `update_policy()`
- [ ] `get_current_policies()`

### query_validator.py
- [ ] `ValidationResult`
- [ ] `QueryValidator`

### task_executor.py
- [ ] `ExecutionResult`
- [ ] `TaskExecutor`
- [ ] `test_task_executor()`

### task_management.py
- [ ] `TaskGraphLogger`
- [ ] `TaskStatus`
- [ ] `TaskType`
- [ ] `Task`
- [ ] `TaskGraph`

### task_plan_parser.py
- [ ] `ParsedPlan`
- [ ] `TaskPlanParser`
- [ ] `test_parser()`

### utils.py
- [ ] `GenomicContext`
- [ ] `safe_log_data()`
- [ ] `setup_debug_logging()`
- [ ] `ResultStreamer`

### whole_genome_reader.py
- [ ] `GeneContext`
- [ ] `ContigContext`
- [ ] `GenomeContext`
- [ ] `WholeGenomeReader`

### __init__.py

### memory_utils.py
- [ ] `generate_session_id()`
- [ ] `ensure_session_directory()`
- [ ] `validate_note_structure()`
- [ ] `save_note_to_file()`
- [ ] `load_note_from_file()`
- [ ] `get_session_stats()`
- [ ] `cleanup_old_sessions()`
- [ ] `search_notes()`
- [ ] `_calculate_relevance()`
- [ ] `_generate_content_summary()`
- [ ] `get_cross_task_connections()`
- [ ] `estimate_storage_usage()`

### model_allocation.py
- [ ] `TaskComplexity`
- [ ] `ModelTier`
- [ ] `ModelConfig`
- [ ] `ModelAllocation`
- [ ] `get_model_allocator()`
- [ ] `switch_to_premium_everywhere()`
- [ ] `switch_to_optimized_mode()`

### model_config.py
- [ ] `ModelConfigManager`
- [ ] `set_optimized_mode()`
- [ ] `set_premium_mode()`
- [ ] `set_testing_mode()`
- [ ] `get_current_mode()`
- [ ] `print_model_status()`
- [ ] `get_config_manager()`
- [ ] `quick_switch_to_o3()`
- [ ] `quick_switch_to_optimized()`
- [ ] `quick_switch_to_testing()`
- [ ] `demo_model_switching()`

### multipart_synthesizer.py
- [ ] `MultiPartReportSynthesizer`

### note_keeper.py
- [ ] `NoteKeeper`

### note_schemas.py
- [ ] `ConnectionType`
- [ ] `ConfidenceLevel`
- [ ] `CrossTaskConnection`
- [ ] `NotingDecisionResult`
- [ ] `TaskNote`
- [ ] `SynthesisNote`
- [ ] `SessionMetadata`
- [ ] `NoteSearchResult`
- [ ] `SessionStats`

### parallel_config.py
- [ ] `ParallelExecutionProfile`
- [ ] `ParallelConfigManager`
- [ ] `set_parallel_profile()`
- [ ] `set_custom_parallel_config()`
- [ ] `get_parallel_config()`
- [ ] `print_parallel_status()`
- [ ] `print_parallel_profiles()`
- [ ] `estimate_parallel_speedup()`
- [ ] `set_conservative_parallel()`
- [ ] `set_balanced_parallel()`
- [ ] `set_aggressive_parallel()`
- [ ] `set_ultra_parallel()`

### parallel_task_executor.py
- [ ] `ParallelExecutionConfig`
- [ ] `ParallelTaskExecutor`
- [ ] `AsyncTaskExecutor`

### progressive_synthesizer.py
- [ ] `ProgressiveSynthesizer`

### report_manager.py
- [ ] `ReportType`
- [ ] `ChunkingStrategy`
- [ ] `ReportChunk`
- [ ] `ReportPlan`
- [ ] `ReportPlanner`

### session_results_accumulator.py
- [ ] `ConfidenceLevel`
- [ ] `DiscoveryType`
- [ ] `ProphageCandidate`
- [ ] `HypotheticalStretch`
- [ ] `OperonPrediction`
- [ ] `SpatialPattern`
- [ ] `SessionResultsAccumulator`

### task_based_synthesizer.py
- [ ] `AnalysisChunk`
- [ ] `TaskBasedSynthesizer`

### test_model_allocation.py
- [ ] `test_model_allocation()`
- [ ] `demo_switching_for_development()`
- [ ] `estimate_cost_comparison()`

### __init__.py

### command_runner.py
- [ ] `CommandRunner`
- [ ] `run_with_file_redirect()`
- [ ] `print_command_result()`
- [ ] `run_annotation_explorer()`
- [ ] `extract_answer_from_file()`
- [ ] `load_and_summarize_results()`

### cli.py
- [ ] `build()`
- [ ] `ask()`
- [ ] `version()`

