from __future__ import annotations

from pathlib import Path
from time import perf_counter
from uuid import uuid4

from tripmind.agents.critic import CriticAgent
from tripmind.agents.intent import IntentAgent
from tripmind.agents.planner import PlannerAgent
from tripmind.checkpoint import JsonCheckpointStore
from tripmind.llm import OpenAIModelClient
from tripmind.memory import JsonMemoryStore
from tripmind.schemas import (
    AgentTrace,
    CritiqueReport,
    IntentResult,
    PlanVariant,
    ReplanDirective,
    TripState,
    WorkflowArtifacts,
    WorkflowRun,
)
from tripmind.travel_data import consume_provider_warnings


class TripMindRuntime:
    """Typed-state runtime for the TripMind agent workflow."""

    def __init__(
        self,
        memory_path: str | Path = ".tripmind_memory.json",
        checkpoint_dir: str | Path = ".tripmind_checkpoints",
        use_llm: bool = True,
        model: str | None = None,
        llm_client: OpenAIModelClient | None = None,
    ) -> None:
        self.memory_store = JsonMemoryStore(memory_path)
        self.checkpoint_store = JsonCheckpointStore(checkpoint_dir)
        self.llm_client = llm_client or (OpenAIModelClient(model=model) if use_llm else None)
        self.intent_agent = IntentAgent(llm=self.llm_client)
        self.planner_agent = PlannerAgent(llm=self.llm_client)
        self.critic_agent = CriticAgent(llm=self.llm_client)

    def run(
        self,
        text: str,
        user_id: str = "demo",
        auto_confirm: bool = True,
        run_id: str | None = None,
    ) -> WorkflowRun:
        run_id = run_id or uuid4().hex
        memory = self.memory_store.get(user_id)
        episodic_hits = self.memory_store.retrieve_episodes(user_id, destination=_guess_destination(text), preferences=[])
        artifacts = WorkflowArtifacts(memory_pack=memory, episodic_memory_hits=episodic_hits, checkpoint_id=run_id)
        trace: list[AgentTrace] = []

        intent, intent_trace = self._collect(text, memory)
        trace.append(intent_trace)
        if intent.request is None or intent.blocking_missing_fields:
            run = WorkflowRun(
                state=TripState.collecting,
                user_id=user_id,
                run_id=run_id,
                intent=intent,
                memory=memory,
                artifacts=artifacts,
                trace=trace,
            )
            self.checkpoint_store.save(run)
            return run

        episodic_hits = self.memory_store.retrieve_episodes(
            user_id,
            destination=intent.request.destination,
            preferences=intent.request.preferences,
        )
        artifacts.episodic_memory_hits = episodic_hits

        run = WorkflowRun(
            state=TripState.collecting,
            user_id=user_id,
            run_id=run_id,
            intent=intent,
            memory=memory,
            artifacts=artifacts,
            trace=trace,
        )
        self.checkpoint_store.save(run)

        if intent.branchable_missing_fields:
            variants, planner_trace, critic_trace = self._plan_variants(intent, memory, episodic_hits)
            trace.extend([planner_trace, critic_trace])
            best = next((variant for variant in variants if variant.recommended), variants[0])
            itinerary = best.itinerary
            critique = best.critique
            artifacts.plan_variants = variants
            artifacts.replan_count = 0
            artifacts = _artifacts_from_itinerary(artifacts, itinerary)
        else:
            itinerary, critique, planner_trace, critic_trace, replan_count = self._plan_with_replan(intent, memory, episodic_hits)
            trace.extend([planner_trace, critic_trace])
            artifacts.replan_count = replan_count
            artifacts = _artifacts_from_itinerary(artifacts, itinerary)

        artifacts.evaluation_metrics = critique.metrics
        for issue in critique.issues:
            itinerary.adjustment_suggestions.append(issue.suggestion)
        run = WorkflowRun(
            state=TripState.awaiting_confirmation,
            user_id=user_id,
            run_id=run_id,
            intent=intent,
            itinerary=itinerary,
            critique=critique,
            memory=memory,
            artifacts=artifacts,
            trace=trace,
        )
        self.checkpoint_store.save(run)

        if not auto_confirm:
            return run

        return self.confirm(run_id)

    def confirm(self, run_id: str) -> WorkflowRun:
        run = self.checkpoint_store.load(run_id)
        if run.intent.request is None:
            return run
        memory = self.memory_store.update_from_request(run.user_id, run.intent.request)
        run.memory = memory
        run.artifacts.memory_pack = memory
        run.state = TripState.confirmed
        self.memory_store.add_episode(run)
        run.trace.append(
            AgentTrace(
                agent="Memory",
                state=TripState.confirmed,
                summary="Saved profile memory and episodic memory after human confirmation.",
                input_schema="TripRequest",
                output_schema="UserMemory + EpisodicMemoryEntry",
            )
        )
        self.checkpoint_store.save(run)
        return run

    def resume(self, run_id: str) -> WorkflowRun:
        return self.checkpoint_store.load(run_id)

    def _collect(self, text: str, memory) -> tuple[IntentResult, AgentTrace]:
        started = perf_counter()
        intent = self.intent_agent.parse(text, memory)
        elapsed_ms = _elapsed_ms(started)
        return intent, AgentTrace(
            agent="IntentAgent",
            state=TripState.collecting,
            summary="Parsed natural language into TripRequest with structured LLM output."
            if self.llm_client
            else "Parsed natural language into TripRequest with deterministic local parser.",
            input_schema="text + UserMemory",
            output_schema="IntentResult",
            metadata=_trace_metadata(self.llm_client, elapsed_ms) if self.llm_client else {"mode": "local", "elapsed_ms": elapsed_ms},
        )

    def _plan(self, intent: IntentResult, memory, directive: ReplanDirective | None = None) -> tuple[object, AgentTrace]:
        started = perf_counter()
        itinerary = self.planner_agent.plan(intent.request, memory, directive=directive)
        elapsed_ms = _elapsed_ms(started)
        summary = "LLM selected from MCP tool candidates; MCP tools calculated transit and budget."
        if directive is not None:
            summary = "Planner refined the itinerary with bounded replan constraints and MCP-backed candidate selection."
        metadata = _trace_metadata(self.llm_client, elapsed_ms) if self.llm_client else {"mode": "local", "elapsed_ms": elapsed_ms}
        if directive is not None:
            metadata["replan_directive"] = directive.model_dump(mode="json")
        return itinerary, AgentTrace(
            agent="PlannerAgent",
            state=TripState.planning,
            summary=summary if self.llm_client else "Generated itinerary using MCP search and estimation tools.",
            input_schema="TripRequest + UserMemory",
            output_schema="Itinerary",
            tool_calls=[
                "mcp://tripmind-travel-tools/attractions_search",
                "mcp://tripmind-travel-tools/restaurant_search",
                "mcp://tripmind-travel-tools/estimate_transit",
                "mcp://tripmind-travel-tools/estimate_budget",
            ],
            metadata=metadata,
        )

    def _review(self, intent: IntentResult, itinerary, memory) -> tuple[object, AgentTrace]:
        started = perf_counter()
        critique = self.critic_agent.review(intent.request, itinerary, memory)
        elapsed_ms = _elapsed_ms(started)
        return critique, AgentTrace(
            agent="CriticAgent",
            state=TripState.reviewing,
            summary="Reviewed itinerary with LLM critique plus deterministic metric checks."
            if self.llm_client
            else "Checked pace, budget, preference coverage, duplicates, and route pressure.",
            input_schema="TripRequest + Itinerary + UserMemory",
            output_schema="CritiqueReport",
            metadata=_trace_metadata(self.llm_client, elapsed_ms) if self.llm_client else {"mode": "local", "elapsed_ms": elapsed_ms},
        )

    def _plan_with_replan(self, intent: IntentResult, memory, episodic_hits) -> tuple[object, CritiqueReport, AgentTrace, AgentTrace, int]:
        itinerary, planner_trace = self._plan(intent, memory)
        critique, critic_trace = self._review(intent, itinerary, memory)
        best_itinerary = itinerary
        best_critique = critique
        replan_count = 0
        if _should_replan(critique):
            directive = _directive_from_critique(intent, itinerary, critique, episodic_hits)
            replanned_itinerary, replanner_trace = self._plan(intent, memory, directive=directive)
            replanned_critique, recritic_trace = self._review(intent, replanned_itinerary, memory)
            replan_count = 1
            if _critique_score(replanned_critique) >= _critique_score(best_critique):
                best_itinerary = replanned_itinerary
                best_critique = replanned_critique
                planner_trace = replanner_trace
                critic_trace = recritic_trace
                planner_trace.metadata["replan"] = True
                critic_trace.metadata["replan"] = True
        return best_itinerary, best_critique, planner_trace, critic_trace, replan_count

    def _plan_variants(self, intent: IntentResult, memory, episodic_hits) -> tuple[list[PlanVariant], AgentTrace, AgentTrace]:
        started = perf_counter()
        variants = self.planner_agent.plan_variants(intent.request, memory)
        elapsed_ms = _elapsed_ms(started)
        plan_variants: list[PlanVariant] = []
        for variant_id, label, itinerary in variants:
            critique = self.critic_agent.review(intent.request, itinerary, memory)
            itinerary.adjustment_suggestions.append(f"{label} 可根据你的真实预算继续细调。")
            plan_variants.append(
                PlanVariant(
                    variant_id=variant_id,
                    label=label,
                    reason=f"Generated because {', '.join(intent.branchable_missing_fields)} was missing.",
                    itinerary=itinerary,
                    critique=critique,
                )
            )
        best_variant = max(plan_variants, key=lambda variant: _critique_score(variant.critique))
        best_variant.recommended = True
        planner_trace = AgentTrace(
            agent="PlannerAgent",
            state=TripState.planning,
            summary="Generated multiple bounded plan variants because branchable fields were missing.",
            input_schema="TripRequest + UserMemory",
            output_schema="list[PlanVariant]",
            tool_calls=[
                "mcp://tripmind-travel-tools/attractions_search",
                "mcp://tripmind-travel-tools/restaurant_search",
                "mcp://tripmind-travel-tools/estimate_transit",
                "mcp://tripmind-travel-tools/estimate_budget",
            ],
            metadata=_trace_metadata(self.llm_client, elapsed_ms) if self.llm_client else {"mode": "local", "elapsed_ms": elapsed_ms},
        )
        critic_trace = AgentTrace(
            agent="CriticAgent",
            state=TripState.reviewing,
            summary="Compared budget-tier variants and selected the strongest version as the recommended plan.",
            input_schema="list[Itinerary]",
            output_schema="list[PlanVariant]",
            metadata={"variant_count": len(plan_variants), "recommended_variant": best_variant.variant_id},
        )
        return plan_variants, planner_trace, critic_trace


def _llm_metadata(llm_client) -> dict:
    if hasattr(llm_client, "metadata"):
        return llm_client.metadata()
    return {"llm_model": getattr(llm_client, "model", "unknown")}


def _trace_metadata(llm_client, elapsed_ms: int) -> dict:
    metadata = _llm_metadata(llm_client)
    metadata["elapsed_ms"] = elapsed_ms
    if hasattr(llm_client, "last_call_info"):
        metadata.update(getattr(llm_client, "last_call_info"))
    return metadata


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


def _artifacts_from_itinerary(artifacts: WorkflowArtifacts, itinerary) -> WorkflowArtifacts:
    artifacts.selected_attractions = [
        item.title for day in itinerary.days for item in day.items if item.kind == "attraction"
    ]
    artifacts.selected_restaurants = [
        item.title for day in itinerary.days for item in day.items if item.kind == "restaurant"
    ]
    artifacts.route_segments = [
        transit.model_dump(mode="json") for day in itinerary.days for transit in day.transit
    ]
    artifacts.budget_summary = itinerary.budget
    artifacts.data_sources = sorted(
        {
            item.source
            for day in itinerary.days
            for item in day.items
            if getattr(item, "source", None)
        }
    )
    artifacts.data_warnings = consume_provider_warnings()
    return artifacts


def _should_replan(critique: CritiqueReport) -> bool:
    if not critique.issues:
        return False
    if any(issue.severity.value == "error" for issue in critique.issues):
        return True
    return critique.metrics.get("cross_area_jump_count", 0) >= 2 or critique.metrics.get("dominant_area_ratio", 1) < 0.45


def _critique_score(critique: CritiqueReport) -> float:
    severity_weights = {"error": -8.0, "warning": -3.0, "info": -1.0}
    score = 100.0
    for issue in critique.issues:
        score += severity_weights.get(issue.severity.value, -1.0)
    score += critique.metrics.get("preference_coverage", 0) * 10
    score -= critique.metrics.get("duplicate_rate", 0) * 8
    score += critique.metrics.get("dominant_area_ratio", 0) * 8
    score -= critique.metrics.get("budget_ratio", 0) * 3 if critique.metrics.get("budget_ratio", 0) > 1 else 0
    score -= critique.metrics.get("cross_area_jump_count", 0) * 2
    return score


def _directive_from_critique(intent: IntentResult, itinerary, critique: CritiqueReport, episodic_hits) -> ReplanDirective:
    all_items = [item for day in itinerary.days for item in day.items]
    area_counts: dict[str, int] = {}
    for item in all_items:
        area_counts[item.area] = area_counts.get(item.area, 0) + 1
    preferred_areas = [area for area, _ in sorted(area_counts.items(), key=lambda item: item[1], reverse=True)[:2]]
    avoid_areas = [area for area in area_counts if area not in preferred_areas]

    blocked_titles: list[str] = []
    preserve_titles = [
        item.title
        for item in sorted(
            all_items,
            key=lambda item: ((item.rating or 0), (item.review_count or 0), -item.estimated_cost),
            reverse=True,
        )
        if item.area in preferred_areas
    ][:3]

    long_leg_threshold = 10.0 if intent.request and intent.request.pace.value == "relaxed" else 12.0
    for day in itinerary.days:
        dominant_area = _dominant_area(day)
        for transit in day.transit:
            if (transit.distance_km or 0) >= long_leg_threshold:
                blocked_titles.append(transit.destination)
        for item in day.items:
            if dominant_area and item.area != dominant_area and critique.metrics.get("cross_area_jump_count", 0) >= 2:
                blocked_titles.append(item.title)

    if any(issue.code == "budget_exceeded" for issue in critique.issues):
        expensive_items = sorted(all_items, key=lambda item: item.estimated_cost, reverse=True)
        blocked_titles.extend(item.title for item in expensive_items[:2] if item.title not in preserve_titles)

    if any(issue.code == "pace_too_full" for issue in critique.issues):
        candidate = next(
            (
                item.title
                for item in sorted(
                    all_items,
                    key=lambda item: (item.kind == "restaurant", item.estimated_cost, item.duration_minutes),
                    reverse=True,
                )
                if item.title not in preserve_titles
            ),
            None,
        )
        if candidate:
            blocked_titles.append(candidate)

    notes = [issue.message for issue in critique.issues[:3]]
    notes.extend(entry.summary for entry in episodic_hits[:1])
    return ReplanDirective(
        label="critic_refined",
        notes=notes,
        blocked_titles=_unique(blocked_titles),
        preserve_titles=_unique(preserve_titles),
        preferred_areas=preferred_areas,
        avoid_areas=avoid_areas[:3],
        required_preferences=intent.request.preferences if intent.request else [],
        prefer_cheaper=any(issue.code == "budget_exceeded" for issue in critique.issues),
        tighter_area_clustering=critique.metrics.get("cross_area_jump_count", 0) >= 2,
        reduce_intensity=any(issue.code == "pace_too_full" for issue in critique.issues),
        max_attractions_per_day=2 if any(issue.code == "pace_too_full" for issue in critique.issues) else None,
        max_leg_km=8.0 if any(issue.code in {"long_route_leg", "cross_area_jumps", "area_scattered"} for issue in critique.issues) else None,
        target_budget=intent.request.budget if intent.request else None,
    )


def _dominant_area(day) -> str | None:
    counts: dict[str, int] = {}
    for item in day.items:
        counts[item.area] = counts.get(item.area, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _guess_destination(text: str) -> str:
    for city in ["上海", "北京", "杭州", "成都", "广州", "深圳", "南京", "苏州"]:
        if city in text:
            return city
    return ""


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result
