from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tripmind.llm import LLMConfigurationError
from tripmind.memory import JsonMemoryStore
from tripmind.renderer import render_markdown
from tripmind.runtime import TripMindRuntime
from tripmind.schemas import UserMemory, WorkflowRun


class PlanRequest(BaseModel):
    text: str = Field(min_length=1)
    user_id: str = "demo"
    auto_confirm: bool = True
    model: str | None = None
    mock: bool = False


class PlanResponse(BaseModel):
    run: WorkflowRun
    markdown: str


def _memory_path() -> Path:
    return Path(os.getenv("TRIPMIND_MEMORY_PATH", ".tripmind_memory.json"))


app = FastAPI(title="TripMind", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "TRIPMIND_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173",
    ).split(","),
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/trips/plan", response_model=PlanResponse)
def plan_trip(payload: PlanRequest) -> PlanResponse:
    use_llm = not payload.mock and os.getenv("TRIPMIND_USE_MOCK", "").lower() not in {"1", "true", "yes"}
    try:
        runtime = TripMindRuntime(memory_path=_memory_path(), use_llm=use_llm, model=payload.model)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    run = runtime.run(text=payload.text, user_id=payload.user_id, auto_confirm=payload.auto_confirm)
    return PlanResponse(run=run, markdown=render_markdown(run))


@app.get("/runs/{run_id}", response_model=PlanResponse)
def get_run(run_id: str) -> PlanResponse:
    runtime = TripMindRuntime(memory_path=_memory_path(), use_llm=False)
    run = runtime.resume(run_id)
    return PlanResponse(run=run, markdown=render_markdown(run))


@app.post("/runs/{run_id}/confirm", response_model=PlanResponse)
def confirm_run(run_id: str) -> PlanResponse:
    runtime = TripMindRuntime(memory_path=_memory_path(), use_llm=False)
    run = runtime.confirm(run_id)
    return PlanResponse(run=run, markdown=render_markdown(run))


@app.get("/memory/{user_id}", response_model=UserMemory)
def get_memory(user_id: str) -> UserMemory:
    return JsonMemoryStore(_memory_path()).get(user_id)
