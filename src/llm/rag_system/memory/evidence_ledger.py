from __future__ import annotations
from typing import Dict, Any
import json, time, os
from pathlib import Path


def append_claim(ledger_path: str, claim: Dict[str, Any]) -> None:
    Path(os.path.dirname(ledger_path)).mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **claim}, ensure_ascii=False) + "\n")

