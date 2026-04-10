from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class ModelAudit:
    model: str
    recommended_role: str
    normalization_policy: str
    leakage_guard: str
    notes: str


def audit_table_to_markdown(audits: List[ModelAudit]) -> str:
    header = "| model | recommended_role | normalization_policy | leakage_guard | notes |\n"
    sep = "|---|---|---|---|---|\n"
    rows = []
    for item in audits:
        row = f"| {item.model} | {item.recommended_role} | {item.normalization_policy} | {item.leakage_guard} | {item.notes} |"
        rows.append(row)
    return header + sep + "\n".join(rows) + "\n"


def audit_table_to_dicts(audits: List[ModelAudit]) -> List[Dict]:
    return [asdict(a) for a in audits]
