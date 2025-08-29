#!/usr/bin/env bash
set -euo pipefail

# Batch validation for GenomicRAG CLI
# - Runs a curated set of prompts
# - Saves stdout to timestamped files under reports/batch_validation/<ts>
# - Requires active environment (e.g., conda activate genome-kg) and running Neo4j

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="reports/batch_validation/${TS}"
mkdir -p "$OUTDIR"

idx=0

while IFS= read -r prompt; do
  [[ -z "$prompt" ]] && continue
  idx=$((idx+1))
  slug=$(echo "$prompt" | tr -dc '[:alnum:] _-' | tr ' ' '_' | tr -s '_' | cut -c1-80)
  outfile="$OUTDIR/${idx}_${slug}.txt"
  echo "[${idx}] Running prompt: $prompt" | tee -a "$OUTDIR/_batch.log"
  # Run and tee to file
  python -m src.cli ask "$prompt" | tee "$outfile"
  echo "[${idx}] Saved: $outfile" | tee -a "$OUTDIR/_batch.log"
  echo
done << 'EOF'
Tell me about the gas fixation capability of this microbiome. Avoid KEGG. Be thorough and search for all the possible gas fixation reactions you can. Show the TASK GRAPH.
Summarize antibiotic resistance genes (resistome) in this metagenome. Show the TASK GRAPH.
Assess sulfur cycling potential (Sox/Dsr/Sqr markers). Avoid KEGG completeness; rely on keyword discovery. Show the TASK GRAPH.
Identify CRISPR-Cas systems and their likely subtypes; include loci where possible. Show the TASK GRAPH.
Summarize the biosynthetic gene cluster (BGC) landscape (NRPS/PKS/terpene/RiPP/siderophore). Show the TASK GRAPH.
Summarize CAZyme families and inferred substrates for this metagenome. Show the TASK GRAPH.
Find evidence for quorum sensing and secretion systems (T1–T6). Prefer PFAM-first keyword discovery with KO corroboration. Show the TASK GRAPH.
Evaluate vitamin/cofactor biosynthesis (cobalamin, biotin, thiamine, folate): presence and likely gaps. Show the TASK GRAPH.
EOF
echo "Batch complete. Outputs in: $OUTDIR"
