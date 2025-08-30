from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import logging


class FileCypherRunner:
    """Execute precompiled Cypher templates from resources/cypher with params.

    This runner avoids any dynamic Cypher generation and uses the existing
    Neo4j driver from the project's query processor.
    """

    def __init__(self, neo4j_driver) -> None:
        self.driver = neo4j_driver
        # Resolve project root from this file location robustly and find templates dir
        base = Path(__file__).resolve()
        candidates = [
            base.parents[4] / "resources" / "cypher",  # project_root/resources/cypher
            base.parents[3] / "resources" / "cypher",  # src/resources/cypher (fallback)
            Path.cwd() / "resources" / "cypher",       # CWD/resources/cypher (last resort)
        ]
        self._tpl_dir = None
        for c in candidates:
            if c.exists():
                self._tpl_dir = c
                break
        if self._tpl_dir is None:
            raise FileNotFoundError(
                f"Cypher templates directory not found. Tried: {', '.join(str(c) for c in candidates)}"
            )
        self.logger = logging.getLogger(__name__)

    def run_template(self, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        cypher_path = self._tpl_dir / name
        if not cypher_path.exists():
            raise FileNotFoundError(f"Cypher template not found: {cypher_path}")

        cypher = cypher_path.read_text(encoding="utf-8")

        with self.driver.session() as session:
            # concise, high-signal instrumentation
            try:
                pkeys = list((params or {}).keys())
                # reduce noise: log at DEBUG level
                self.logger.debug(f"DB_TEMPLATE_EXECUTE: name={name} param_keys={pkeys}")
            except Exception:
                pass
            result = session.run(cypher, params)
            return [dict(record) for record in result]
