from tripmind.renderer import render_markdown
from tripmind.runtime import TripMindRuntime
from tripmind.schemas import TripState


def test_runtime_completes_confirmed_workflow(tmp_path) -> None:
    runtime = TripMindRuntime(memory_path=tmp_path / "memory.json", checkpoint_dir=tmp_path / "checkpoints", use_llm=False)
    run = runtime.run("去杭州2天，预算900元，喜欢博物馆和自然，轻松一点", user_id="alice")

    assert run.state == TripState.confirmed
    assert run.run_id is not None
    assert run.intent.request is not None
    assert run.itinerary is not None
    assert run.critique is not None
    assert [step.agent for step in run.trace] == ["IntentAgent", "PlannerAgent", "CriticAgent", "Memory"]
    assert run.artifacts.selected_attractions
    assert run.artifacts.budget_summary is not None
    assert "preference_coverage" in run.artifacts.evaluation_metrics
    assert run.artifacts.episodic_memory_hits == []

    markdown = render_markdown(run)
    assert "# TripMind Itinerary: 杭州" in markdown
    assert "## Budget Estimate" in markdown
    assert "## Agent Workflow" in markdown


def test_memory_is_isolated_by_user_id(tmp_path) -> None:
    runtime = TripMindRuntime(memory_path=tmp_path / "memory.json", checkpoint_dir=tmp_path / "checkpoints", use_llm=False)
    runtime.run("去上海2天，预算700元，喜欢博物馆，不要太赶", user_id="alice")
    bob = runtime.memory_store.get("bob")

    assert bob.likes == []
    assert bob.user_id == "bob"


def test_review_only_stops_before_confirmation(tmp_path) -> None:
    runtime = TripMindRuntime(memory_path=tmp_path / "memory.json", checkpoint_dir=tmp_path / "checkpoints", use_llm=False)
    run = runtime.run("去北京1天，预算300元，喜欢历史", user_id="alice", auto_confirm=False)

    assert run.state == TripState.awaiting_confirmation
    assert runtime.memory_store.get("alice").likes == []

    resumed = runtime.resume(run.run_id)
    assert resumed.state == TripState.awaiting_confirmation

    confirmed = runtime.confirm(run.run_id)
    assert confirmed.state == TripState.confirmed
    assert runtime.memory_store.get("alice").likes == ["history"]
    assert runtime.memory_store.retrieve_episodes("alice", destination="北京", preferences=["history"])


def test_runtime_blocks_when_required_fields_are_missing(tmp_path) -> None:
    runtime = TripMindRuntime(memory_path=tmp_path / "memory.json", checkpoint_dir=tmp_path / "checkpoints", use_llm=False)
    run = runtime.run("周末两天，预算500，喜欢美食", user_id="alice", auto_confirm=False)

    assert run.state == TripState.collecting
    assert run.intent.request is None
    assert "destination" in run.intent.blocking_missing_fields
    assert run.intent.clarification_questions


def test_runtime_generates_plan_variants_when_budget_missing(tmp_path) -> None:
    runtime = TripMindRuntime(memory_path=tmp_path / "memory.json", checkpoint_dir=tmp_path / "checkpoints", use_llm=False)
    run = runtime.run("去杭州2天，喜欢博物馆和自然，轻松一点", user_id="alice", auto_confirm=False)

    assert run.state == TripState.awaiting_confirmation
    assert run.artifacts.plan_variants
    assert any(variant.recommended for variant in run.artifacts.plan_variants)
