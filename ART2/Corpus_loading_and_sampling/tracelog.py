"""
TraceLog -- an auditable record of pipeline decisions.

This implements the central requirement of the methodology: that every
decision, from corpus to indicator, is DOCUMENTED, JUSTIFIED, and TRACEABLE.

For each operation it stores:
  - step          : which phase of the pathway it belongs to (1..5)
  - operation     : what was done (name of the function/action)
  - parameters    : with which parameters (the automatic part)
  - justification : why (free text written by the researcher; the code
                    cannot infer it)
  - summary       : facts about the result (e.g. number of units obtained)
  - artifact      : path of the file where the result was saved, if any
  - timestamp     : when

The full log is serialized to JSON, so the entire chain of decisions of a
study can be audited.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class TraceLog:
    """Accumulates the pipeline decisions in order."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def record(self, step: int, operation: str,
               parameters: Optional[Dict[str, Any]] = None,
               justification: str = "",
               summary: Optional[Dict[str, Any]] = None,
               artifact: Optional[str] = None) -> None:
        """Add an entry to the record.

        'justification' is the *why* of the decision: it is worth filling in
        every time, because it is what makes the log auditable in the sense of
        the methodology (knowing what was done is not enough; we need to know
        why).
        """
        self._entries.append({
            "step": step,
            "operation": operation,
            "parameters": parameters or {},
            "justification": justification,
            "summary": summary or {},
            "artifact": artifact,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    def entries(self) -> List[Dict[str, Any]]:
        """Return a copy of the recorded entries."""
        return [dict(e) for e in self._entries]

    def save(self, path: str) -> None:
        """Serialize the full record to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._entries)
