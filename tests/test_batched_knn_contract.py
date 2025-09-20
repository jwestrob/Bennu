import asyncio
from src.llm.vector.lancedb_ops import batched_knn_and_filter


class MockLance:
    def __init__(self, manifest=None):
        self.manifest = manifest or type("M", (), {"dim": 320, "version": "v1"})()

    async def execute_similarity_batch(self, ids, k):
        # Contract: return mapping seed_id -> list of dicts with protein_id and distance
        out = {}
        for sid in ids:
            out[sid] = [
                {"protein_id": f"{sid}_nbr1", "distance": 0.1},
                {"protein_id": f"{sid}_nbr2", "distance": 0.2},
            ]
        return type("QR", (), {"results": [out]})()


class MockDB:
    def run_template(self, name, params):
        assert name.endswith("pfam_flags_for_protein_ids.cypher")
        # Contract: do not fabricate biology; flag none as marker
        return [{"protein_id": pid, "is_marker": False} for pid in params["protein_ids"]]


def test_batched_knn_and_filter_calls_once():
    ldb = MockLance()
    db = MockDB()
    out = asyncio.run(
        batched_knn_and_filter(
            lancedb_processor=ldb,
            seed_ids=["A", "B"],
            topk=20,
            distance="cosine",
            pfam_filter_ids=["PF00589"],
            pfam_filter_needle="integrase",
            neo4j_runner=db,
        )
    )
    assert set(out.keys()) == {"A", "B"}
    assert len(out["A"]) >= 2

