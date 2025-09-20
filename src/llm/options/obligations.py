from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ObligationLedger:
    state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_intent(cls, intent) -> "ObligationLedger":
        ledger = cls()
        # seed selection and neighborhoods are implicit in fast path; mark as pending
        ledger.state["seed_selection"] = {"required": True, "done": False}
        ledger.state["neighborhoods"] = {
            "required": True,
            "done": False,
            "min_k": intent.flank.value,
        }
        # LanceDB KNN
        if getattr(intent.obligations, "lancedb_knn", None) and intent.obligations.lancedb_knn.required:
            ldb = intent.obligations.lancedb_knn
            ledger.state["lancedb_knn"] = {
                "required": True,
                "done": False,
                "nn": ldb.nn,
                "exclude_markers": ldb.exclude_markers,
                "exclude_namespace": ldb.exclude_namespace,
                "distance": ldb.distance,
            }
        else:
            ledger.state["lancedb_knn"] = {"required": False, "done": True}
        # Literature
        ledger.state["literature"] = {
            "required": bool(getattr(intent.obligations, "literature", False)),
            "done": not bool(getattr(intent.obligations, "literature", False)),
        }
        return ledger

    def mark_done(self, name: str):
        if name in self.state:
            self.state[name]["done"] = True

    def unmet(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self.state.items() if v.get("required") and not v.get("done")}

