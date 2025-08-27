import os
import pytest


def _neo4j_available():
    try:
        from src.llm.query_processor import Neo4jQueryProcessor
        from src.llm.config import LLMConfig
        cfg = LLMConfig()
        proc = Neo4jQueryProcessor(cfg)
        # Try a lightweight ping
        with proc.driver.session() as s:
            s.run("RETURN 1 AS ok").single()
        return proc
    except Exception:
        return None


@pytest.mark.skipif(os.getenv("RUN_NEO4J_TEMPLATE_TESTS", "0") != "1", reason="Set RUN_NEO4J_TEMPLATE_TESTS=1 to enable DB tests")
def test_present_kos_by_genome_explain():
    proc = _neo4j_available()
    if proc is None:
        pytest.skip("Neo4j not available")
    from src.llm.options.template_runner import FileCypherRunner
    r = FileCypherRunner(proc.driver)
    name = "present_kos_by_genome.cypher"
    cypher = (r._tpl_dir / name).read_text()
    with proc.driver.session() as s:
        s.run("EXPLAIN " + cypher, {"genome_ids": []}).consume()


@pytest.mark.skipif(os.getenv("RUN_NEO4J_TEMPLATE_TESTS", "0") != "1", reason="Set RUN_NEO4J_TEMPLATE_TESTS=1 to enable DB tests")
def test_bgcs_by_genome_explain():
    proc = _neo4j_available()
    if proc is None:
        pytest.skip("Neo4j not available")
    from src.llm.options.template_runner import FileCypherRunner
    r = FileCypherRunner(proc.driver)
    name = "bgcs_by_genome.cypher"
    cypher = (r._tpl_dir / name).read_text()
    params = {
        "genome_id": None,
        "genome_ids": [],
        "id_keys": ["bgcId", "bgc_id", "id"],
        "product_keys": ["bgcProduct", "bgc_product", "product", "cluster_type"],
        "contig_keys": ["contig", "scaffold", "seqid"],
        "start_keys": ["startCoordinate", "start", "begin", "start_position"],
        "end_keys": ["endCoordinate", "end", "finish", "end_position"],
        "length_keys": ["lengthNt", "length"],
        "protein_keys": ["proteinCount", "proteins", "protein_count"],
        "avg_prob_keys": ["averageProbability", "avg_probability", "average_p"],
        "max_prob_keys": ["maxProbability", "max_probability", "max_p"],
    }
    with proc.driver.session() as s:
        s.run("EXPLAIN " + cypher, params).consume()


@pytest.mark.skipif(os.getenv("RUN_NEO4J_TEMPLATE_TESTS", "0") != "1", reason="Set RUN_NEO4J_TEMPLATE_TESTS=1 to enable DB tests")
def test_cazymes_by_genome_explain():
    proc = _neo4j_available()
    if proc is None:
        pytest.skip("Neo4j not available")
    from src.llm.options.template_runner import FileCypherRunner
    r = FileCypherRunner(proc.driver)
    name = "cazymes_by_genome.cypher"
    cypher = (r._tpl_dir / name).read_text()
    with proc.driver.session() as s:
        s.run("EXPLAIN " + cypher, {"genome_id": None, "genome_ids": []}).consume()


@pytest.mark.skipif(os.getenv("RUN_NEO4J_TEMPLATE_TESTS", "0") != "1", reason="Set RUN_NEO4J_TEMPLATE_TESTS=1 to enable DB tests")
def test_keyword_protein_queries_explain():
    proc = _neo4j_available()
    if proc is None:
        pytest.skip("Neo4j not available")
    from src.llm.options.template_runner import FileCypherRunner
    r = FileCypherRunner(proc.driver)
    for name in ("proteins_by_pfam_keyword.cypher", "proteins_by_ko_keyword.cypher"):
        cypher = (r._tpl_dir / name).read_text()
        with proc.driver.session() as s:
            s.run("EXPLAIN " + cypher, {"q": "hydrogenase", "limit": 10, "genome_ids": []}).consume()

