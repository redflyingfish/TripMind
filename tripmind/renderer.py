from __future__ import annotations

from tripmind.schemas import IssueSeverity, WorkflowRun


def render_markdown(run: WorkflowRun) -> str:
    if run.intent.request is None:
        missing = ", ".join(run.intent.blocking_missing_fields or run.intent.missing_fields) or "unknown"
        notes = "\n".join(f"- {note}" for note in run.intent.notes)
        questions = "\n".join(f"- {question}" for question in run.intent.clarification_questions)
        return (
            "# TripMind needs more information\n\n"
            f"Missing fields: {missing}\n\n"
            "## Questions\n"
            f"{questions or '- Please provide the missing fields.'}\n\n"
            f"{notes}\n"
        )

    request = run.intent.request
    if run.itinerary is None and run.artifacts.plan_variants:
        lines = [
            f"# TripMind Variants: {request.destination}",
            "",
            f"- State: `{run.state.value}`",
            f"- Missing but branchable fields: {', '.join(run.intent.branchable_missing_fields) or 'none'}",
            "",
            "## Recommended Variants",
        ]
        for variant in run.artifacts.plan_variants:
            marker = " (recommended)" if variant.recommended else ""
            lines.append(
                f"- **{variant.label}{marker}**: total ~{variant.itinerary.budget.total:.0f} "
                f"{variant.itinerary.budget.currency}; issues={len(variant.critique.issues)}"
            )
        return "\n".join(lines) + "\n"

    if run.itinerary is None:
        return "# TripMind\n\nNo itinerary available.\n"

    itinerary = run.itinerary
    lines: list[str] = [
        f"# TripMind Itinerary: {request.destination}",
        "",
        f"- State: `{run.state.value}`",
        f"- Duration: {request.days} day(s)",
        f"- Pace: {request.pace.value}",
        f"- Budget: {request.budget or 'not provided'} {request.currency}",
        f"- Preferences: {', '.join(request.preferences) or 'not provided'}",
        "",
        "## Daily Plan",
    ]

    if run.intent.branchable_missing_fields:
        lines.extend(
            [
                "",
                "## Branchable Missing Fields",
                f"- {', '.join(run.intent.branchable_missing_fields)}",
                "- TripMind generated plan variants because these fields were missing.",
            ]
        )

    for day in itinerary.days:
        lines.extend(["", f"### Day {day.day}: {day.theme}"])
        for item in day.items:
            meta: list[str] = [item.kind, item.area, f"{item.duration_minutes} min", f"~{item.estimated_cost:.0f} {request.currency}"]
            if item.source:
                meta.append(f"source: {item.source}")
            if item.rating is not None:
                meta.append(f"rating: {item.rating:.1f}")
            if item.review_count is not None:
                meta.append(f"reviews: {item.review_count}")
            lines.append(
                f"- **{_display_time_of_day(item.time_of_day)}** · {item.title} ({', '.join(meta)})"
            )
        if day.transit:
            transit_minutes = sum(transit.minutes for transit in day.transit)
            lines.append(f"- Transit: about {transit_minutes} min total")
        lines.append(f"- Day estimate: ~{day.estimated_cost:.0f} {request.currency}")

    budget = itinerary.budget
    lines.extend(
        [
            "",
            "## Budget Estimate",
            f"- Attractions: {budget.attractions:.0f} {budget.currency}",
            f"- Restaurants: {budget.restaurants:.0f} {budget.currency}",
            f"- Local transit: {budget.transit:.0f} {budget.currency}",
            f"- Buffer: {budget.buffer:.0f} {budget.currency}",
            f"- **Total: {budget.total:.0f} {budget.currency}**",
            "",
            "## Risks And Review",
        ]
    )

    if run.critique and run.critique.issues:
        for issue in run.critique.issues:
            label = "Risk" if issue.severity in {IssueSeverity.warning, IssueSeverity.error} else "Note"
            lines.append(f"- **{label} ({issue.code})**: {issue.message} Suggestion: {issue.suggestion}")
    else:
        lines.append("- No major issues found by CriticAgent.")

    if run.artifacts.evaluation_metrics:
        lines.extend(["", "## Evaluation Metrics"])
        for name, value in run.artifacts.evaluation_metrics.items():
            lines.append(f"- {name}: {value}")

    if run.artifacts.selected_attractions or run.artifacts.selected_restaurants:
        lines.extend(["", "## Artifacts"])
        if run.artifacts.selected_attractions:
            lines.append(f"- selected_attractions: {', '.join(run.artifacts.selected_attractions)}")
        if run.artifacts.selected_restaurants:
            lines.append(f"- selected_restaurants: {', '.join(run.artifacts.selected_restaurants)}")
        if run.artifacts.data_sources:
            lines.append(f"- data_sources: {', '.join(run.artifacts.data_sources)}")
        if run.artifacts.data_warnings:
            lines.append(f"- data_warnings: {' | '.join(run.artifacts.data_warnings)}")
        if run.artifacts.replan_count:
            lines.append(f"- replan_count: {run.artifacts.replan_count}")
        if run.run_id:
            lines.append(f"- checkpoint_id: {run.run_id}")

    if run.artifacts.plan_variants:
        lines.extend(["", "## Plan Variants"])
        for variant in run.artifacts.plan_variants:
            marker = " (recommended)" if variant.recommended else ""
            lines.append(
                f"- {variant.label}{marker}: ~{variant.itinerary.budget.total:.0f} "
                f"{variant.itinerary.budget.currency}, issues={len(variant.critique.issues)}"
            )

    if run.artifacts.episodic_memory_hits:
        lines.extend(["", "## Episodic Memory"])
        for entry in run.artifacts.episodic_memory_hits:
            lines.append(f"- {entry.summary}")

    lines.extend(["", "## Adjustment Suggestions"])
    suggestions = _unique(itinerary.adjustment_suggestions)
    if suggestions:
        lines.extend(f"- {suggestion}" for suggestion in suggestions)
    else:
        lines.append("- Keep the plan as-is, or ask TripMind to make it slower, cheaper, or more food-focused.")

    lines.extend(["", "## Assumptions"])
    lines.extend(f"- {assumption}" for assumption in itinerary.assumptions)

    lines.extend(["", "## Agent Workflow"])
    for step in run.trace:
        tools = f" Tools: {', '.join(step.tool_calls)}." if step.tool_calls else ""
        lines.append(f"- `{step.state.value}` {step.agent}: {step.summary}{tools}")

    return "\n".join(lines) + "\n"


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _display_time_of_day(value: str) -> str:
    mapping = {
        "Breakfast": "早餐",
        "Morning": "上午行程",
        "Lunch": "中餐",
        "Afternoon": "下午行程",
        "Late afternoon": "傍晚前行程",
        "Dinner": "晚餐",
        "Evening": "晚上行程",
    }
    return mapping.get(value, value)
