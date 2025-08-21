from src.llm.rag_system.agent.tools.validate import validate_toolcall


def test_schema_valid_database_query():
    obj = {
        "tool": "database_query",
        "params": {
            "template": "protein_by_id",
            "slots": {"id": "P12345"}
        }
    }
    ok, errs = validate_toolcall(obj)
    assert ok, f"unexpected validation errors: {errs}"


def test_schema_rejects_unknown_fields():
    obj = {
        "tool": "database_query",
        "params": {
            "template": "protein_by_id",
            "slots": {"id": "P12345"},
            "extra": 1
        }
    }
    ok, errs = validate_toolcall(obj)
    assert not ok

