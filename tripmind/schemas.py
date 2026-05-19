from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Pace(str, Enum):
    relaxed = "relaxed"
    balanced = "balanced"
    packed = "packed"


class TripState(str, Enum):
    collecting = "collecting"
    planning = "planning"
    reviewing = "reviewing"
    awaiting_confirmation = "awaiting_confirmation"
    confirmed = "confirmed"


class IssueSeverity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class TripRequest(BaseModel):
    destination: str
    start_date: date | None = None
    days: int = Field(default=2, ge=1, le=14)
    budget: float | None = Field(default=None, ge=0)
    currency: str = "CNY"
    travelers: int = Field(default=1, ge=1, le=12)
    preferences: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    requested_time_slots: list[str] = Field(default_factory=list)
    avoid_time_slots: list[str] = Field(default_factory=list)
    pace: Pace = Pace.balanced
    raw_text: str = ""


class IntentResult(BaseModel):
    request: TripRequest | None = None
    missing_fields: list[str] = Field(default_factory=list)
    blocking_missing_fields: list[str] = Field(default_factory=list)
    branchable_missing_fields: list[str] = Field(default_factory=list)
    advisory_missing_fields: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.request is None or bool(self.blocking_missing_fields)


class Attraction(BaseModel):
    name: str
    area: str
    tags: list[str]
    duration_minutes: int
    ticket_price: float
    crowd_level: int = Field(ge=1, le=5)
    source: str | None = None
    provider_note: str | None = None
    rating: float | None = None
    review_count: int | None = None
    latitude: float | None = None
    longitude: float | None = None


class Restaurant(BaseModel):
    name: str
    area: str
    tags: list[str]
    average_price: float
    source: str | None = None
    provider_note: str | None = None
    rating: float | None = None
    review_count: int | None = None
    latitude: float | None = None
    longitude: float | None = None


class TransitEstimate(BaseModel):
    origin: str
    destination: str
    minutes: int
    cost: float
    mode: str
    distance_km: float | None = None


class BudgetBreakdown(BaseModel):
    attractions: float = 0
    restaurants: float = 0
    transit: float = 0
    buffer: float = 0
    total: float = 0
    currency: str = "CNY"


class ItineraryItem(BaseModel):
    time_of_day: str
    title: str
    kind: str
    area: str
    duration_minutes: int
    estimated_cost: float
    notes: str = ""
    source: str | None = None
    provider_note: str | None = None
    rating: float | None = None
    review_count: int | None = None
    latitude: float | None = None
    longitude: float | None = None


class DayPlan(BaseModel):
    day: int
    theme: str
    items: list[ItineraryItem] = Field(default_factory=list)
    transit: list[TransitEstimate] = Field(default_factory=list)
    estimated_cost: float = 0


class Itinerary(BaseModel):
    request: TripRequest
    days: list[DayPlan]
    budget: BudgetBreakdown
    assumptions: list[str] = Field(default_factory=list)
    adjustment_suggestions: list[str] = Field(default_factory=list)


class PlannedStop(BaseModel):
    time_of_day: str
    name: str
    kind: Literal["attraction", "restaurant"]
    rationale: str = ""


class DayPlanDraft(BaseModel):
    day: int
    theme: str
    stops: list[PlannedStop] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)


class ItineraryDraft(BaseModel):
    days: list[DayPlanDraft] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    adjustment_suggestions: list[str] = Field(default_factory=list)


class CritiqueIssue(BaseModel):
    severity: IssueSeverity
    code: str
    message: str
    suggestion: str


class CritiqueReport(BaseModel):
    issues: list[CritiqueIssue] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class ReplanDirective(BaseModel):
    label: str = "refined"
    notes: list[str] = Field(default_factory=list)
    blocked_titles: list[str] = Field(default_factory=list)
    preserve_titles: list[str] = Field(default_factory=list)
    preferred_areas: list[str] = Field(default_factory=list)
    avoid_areas: list[str] = Field(default_factory=list)
    required_preferences: list[str] = Field(default_factory=list)
    prefer_cheaper: bool = False
    tighter_area_clustering: bool = False
    reduce_intensity: bool = False
    max_attractions_per_day: int | None = None
    max_leg_km: float | None = None
    target_budget: float | None = None


class UserMemory(BaseModel):
    user_id: str
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    preferred_pace: Pace | None = None
    budget_sensitive: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodicMemoryEntry(BaseModel):
    destination: str
    summary: str
    days: int
    budget: float | None = None
    pace: Pace | None = None
    preferences: list[str] = Field(default_factory=list)
    selected_titles: list[str] = Field(default_factory=list)
    selected_areas: list[str] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentTrace(BaseModel):
    agent: str
    state: TripState
    summary: str
    input_schema: str
    output_schema: str
    tool_calls: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowArtifacts(BaseModel):
    selected_attractions: list[str] = Field(default_factory=list)
    selected_restaurants: list[str] = Field(default_factory=list)
    route_segments: list[dict[str, Any]] = Field(default_factory=list)
    budget_summary: BudgetBreakdown | None = None
    memory_pack: UserMemory | None = None
    episodic_memory_hits: list[EpisodicMemoryEntry] = Field(default_factory=list)
    evaluation_metrics: dict[str, float] = Field(default_factory=dict)
    data_sources: list[str] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)
    plan_variants: list["PlanVariant"] = Field(default_factory=list)
    replan_count: int = 0
    checkpoint_id: str | None = None


class WorkflowRun(BaseModel):
    state: TripState
    user_id: str
    run_id: str | None = None
    intent: IntentResult
    itinerary: Itinerary | None = None
    critique: CritiqueReport | None = None
    memory: UserMemory
    artifacts: WorkflowArtifacts = Field(default_factory=WorkflowArtifacts)
    trace: list[AgentTrace] = Field(default_factory=list)


class PlanVariant(BaseModel):
    variant_id: str
    label: str
    reason: str = ""
    itinerary: Itinerary
    critique: CritiqueReport
    recommended: bool = False
