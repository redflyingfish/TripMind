from __future__ import annotations

import json
from pathlib import Path

from tripmind.schemas import WorkflowRun


class JsonCheckpointStore:
    """Small checkpoint store for resumable agent runs."""

    def __init__(self, directory: str | Path = ".tripmind_checkpoints") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, run: WorkflowRun) -> None:
        if not run.run_id:
            return
        path = self.directory / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def load(self, run_id: str) -> WorkflowRun:
        path = self.directory / f"{run_id}.json"
        return WorkflowRun.model_validate_json(path.read_text(encoding="utf-8"))

