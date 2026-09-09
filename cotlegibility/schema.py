"""One JSONL row per message; exact text and provenance survive processing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass
class Sample:
    id: str
    provider: str
    model: str
    channel: str
    text: str
    source: str
    timestamp: str | None = None
    session_id: str | None = None
    context_id: str | None = None
    status: str = "ok"  # ok, unobservable, missing, error
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or not isinstance(self.text, str):
            raise ValueError("Samples need a nonempty id and string text")
        if self.status not in {"ok", "unobservable", "missing", "error"}:
            raise ValueError(f"Unknown sample status: {self.status}")
        self.meta = self.meta or {}

    def to_dict(self):
        return asdict(self)


def unique_samples(samples):
    """Collapse copies of the same event ID, never repeated text across experiments."""
    seen = {}
    for sample in samples:
        row = sample.to_dict()
        if sample.id in seen:
            if seen[sample.id] != row:
                raise ValueError(f"Conflicting records for sample {sample.id}")
            continue
        seen[sample.id] = row
        yield sample
