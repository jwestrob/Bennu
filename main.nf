#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Pipeline parameters
params.input_dir = "data/raw"
params.outdir = "results"
params.min_contig_length = 500
params.genetic_code = 11
params.threads = 4
params.copy_files = false
params.skip_taxonomy = false
params.enable_cc = false

// Import modules
include { PREPARE_INPUTS } from './modules/prepare_inputs.nf'
include { QUAST_QC } from './modules/quast.nf' 
include { DFAST_QC } from './modules/dfast_qc.nf'
include { PRODIGAL } from './modules/prodigal.nf'
include { ASTRA_SCAN } from './modules/astra.nf'
include { BUILD_KNOWLEDGE_GRAPH } from './modules/build_kg.nf'
include { CREATE_LLM_INDICES } from './modules/llm_setup.nf'

workflow GENOME_TO_LLM_KG {
    
    // Stage 0: Input preparation and validation
    input_genomes = Channel.fromPath("${params.input_dir}/*.{fna,fasta,fa}")
        .collect()
    
    PREPARE_INPUTS(input_genomes)
    
    // Create genome ID channel from prepared inputs
    genome_ch = PREPARE_INPUTS.out.genomes
        .flatten()
        .map { file -> 
            def genome_id = file.baseName.replaceAll(/\.(fna|fasta|fa)$/, '')
            tuple(genome_id, file)
        }
    
    // Stage 1: Quality assessment with QUAST
    QUAST_QC(genome_ch)
    
    // Stage 2: Taxonomic classification (optional)
    if (!params.skip_taxonomy) {
        DFAST_QC(genome_ch)
        taxonomy_ch = DFAST_QC.out.results
    } else {
        taxonomy_ch = Channel.empty()
    }
    
    // Stage 3: Gene prediction with Prodigal  
    PRODIGAL(genome_ch)
    
    // Stage 4: Functional annotation
    ASTRA_SCAN(PRODIGAL.out.proteins)
    
    // Combine all results for knowledge graph construction
    kg_input_ch = QUAST_QC.out.results
        .join(taxonomy_ch, remainder: true)
        .join(PRODIGAL.out.results)
        .join(ASTRA_SCAN.out.results)
        .collect()
    
    // Knowledge graph construction
    BUILD_KNOWLEDGE_GRAPH(kg_input_ch)
    
    // LLM index creation
    CREATE_LLM_INDICES(
        BUILD_KNOWLEDGE_GRAPH.out.rdf_files,
        BUILD_KNOWLEDGE_GRAPH.out.neo4j_data,
        PRODIGAL.out.all_proteins.collect()
    )
}

workflow {
    GENOME_TO_LLM_KG()
}

// Workflow completion notification
workflow.onComplete {
    log.info """
    Pipeline completed!
    
    Results published to: ${params.outdir}
    Knowledge graph: ${params.outdir}/knowledge_graph/
    LLM indices: ${params.outdir}/llm_indices/
    
    Duration: ${workflow.duration}
    Success: ${workflow.success}
    """
}