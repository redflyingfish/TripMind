from __future__ import annotations

from tripmind.llm import OpenAIModelClient
from tripmind.schemas import CritiqueIssue, CritiqueReport, IssueSeverity, Itinerary, Pace, TripRequest, UserMemory


class CriticAgent:
    """Review itinerary quality against user constraints and memory."""

    def __init__(self, llm: OpenAIModelClient | None = None) -> None:
        self.llm = llm

    def review(self, request: TripRequest, itinerary: Itinerary, memory: UserMemory) -> CritiqueReport:
        issues: list[CritiqueIssue] = []
        if self.llm:
            issues.extend(self._review_with_llm(request, itinerary, memory).issues)
        self._check_slot_semantics(itinerary, issues)
        self._check_pace(request, itinerary, issues)
        self._check_route_cohesion(request, itinerary, issues)
        self._check_budget(request, itinerary, memory, issues)
        self._check_preferences(request, itinerary, memory, issues)
        issues = _filter_budget_false_positives(request, itinerary, issues)
        issues = _dedupe_issues(issues)
        passed = all(issue.severity != IssueSeverity.error for issue in issues)
        return CritiqueReport(issues=issues, metrics=_evaluation_metrics(request, itinerary, issues), passed=passed)

    def _review_with_llm(self, request: TripRequest, itinerary: Itinerary, memory: UserMemory) -> CritiqueReport:
        system = _critic_system_prompt()
        user = (
            f"TripRequest:\n{request.model_dump_json(exclude={'raw_text'}, exclude_none=True)}\n\n"
            f"UserMemory:\n{memory.model_dump_json(exclude={'updated_at'}, exclude_none=True)}\n\n"
            f"Itinerary:\n{itinerary.model_dump_json(exclude={'request'}, exclude_none=True)}\n\n"
            "请返回 CritiqueReport。"
        )
        return self.llm.parse(system=system, user=user, schema=CritiqueReport)

    def _check_pace(self, request: TripRequest, itinerary: Itinerary, issues: list[CritiqueIssue]) -> None:
        max_minutes = {Pace.relaxed: 390, Pace.balanced: 520, Pace.packed: 650}[request.pace]
        for day in itinerary.days:
            scheduled = sum(item.duration_minutes for item in day.items) + sum(transit.minutes for transit in day.transit)
            if scheduled > max_minutes:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.warning,
                        code="pace_too_full",
                        message=f"Day {day.day} has about {scheduled} scheduled minutes.",
                        suggestion="Remove one stop or make dinner close to the afternoon area.",
                    )
                )
            long_transits = [transit for transit in day.transit if transit.minutes >= 36]
            if len(long_transits) >= 2:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.info,
                        code="route_has_transfers",
                        message=f"Day {day.day} includes several cross-area transfers.",
                        suggestion="Group attractions by area if the user wants a calmer route.",
                    )
                )

    def _check_budget(
        self,
        request: TripRequest,
        itinerary: Itinerary,
        memory: UserMemory,
        issues: list[CritiqueIssue],
    ) -> None:
        if request.budget is not None and itinerary.budget.total > request.budget:
            issues.append(
                CritiqueIssue(
                    severity=IssueSeverity.error,
                    code="budget_exceeded",
                    message=f"Estimated cost {itinerary.budget.total:.0f} {request.currency} exceeds budget {request.budget:.0f} {request.currency}.",
                    suggestion="Swap paid attractions for free citywalks and reduce restaurant spend.",
                )
            )
        elif request.budget is None and memory.budget_sensitive:
            issues.append(
                CritiqueIssue(
                    severity=IssueSeverity.info,
                    code="budget_missing",
                    message="User memory says they are budget sensitive, but this request has no budget.",
                    suggestion="Ask for a budget before booking-like decisions.",
                )
            )

    def _check_slot_semantics(self, itinerary: Itinerary, issues: list[CritiqueIssue]) -> None:
        for day in itinerary.days:
            slots = {_normalize_time_of_day(item.time_of_day): item for item in day.items}
            for slot in ("Lunch", "Dinner"):
                item = slots.get(slot)
                if item is None:
                    issues.append(
                        CritiqueIssue(
                            severity=IssueSeverity.warning,
                            code="meal_slot_missing",
                            message=f"Day {day.day} is missing {slot}.",
                            suggestion="Add a restaurant for the meal slot, or explicitly explain why it is omitted.",
                        )
                    )
                elif item.kind != "restaurant":
                    issues.append(
                        CritiqueIssue(
                            severity=IssueSeverity.error,
                            code="meal_slot_not_restaurant",
                            message=f"Day {day.day} {slot} is assigned to {item.title}, which is not a restaurant.",
                            suggestion="Replace the meal slot with a restaurant candidate.",
                        )
                    )

            for item in day.items:
                slot = _normalize_time_of_day(item.time_of_day)
                if slot in {"Morning", "Afternoon", "Late afternoon", "Evening"} and item.kind != "attraction":
                    issues.append(
                        CritiqueIssue(
                            severity=IssueSeverity.error,
                            code="activity_slot_not_attraction",
                            message=f"Day {day.day} {slot} is assigned to {item.title}, which is not an attraction.",
                            suggestion="Move restaurants to Lunch/Dinner and use an attraction for activity slots.",
                        )
                    )

    def _check_route_cohesion(
        self,
        request: TripRequest,
        itinerary: Itinerary,
        issues: list[CritiqueIssue],
    ) -> None:
        for day in itinerary.days:
            area_counts = _area_counts(day)
            dominant_ratio = _dominant_area_ratio(area_counts, len(day.items))
            cross_area_jump_count = _cross_area_jump_count(day)
            max_leg_km = _max_leg_km(day)
            far_outlier_count = _far_outlier_stop_count(day)

            relaxed_ratio_floor = 0.6
            balanced_ratio_floor = 0.45
            if request.pace == Pace.relaxed and dominant_ratio < relaxed_ratio_floor:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.warning,
                        code="area_scattered",
                        message=f"Day {day.day} is spread across too many areas for a relaxed plan (dominant area ratio {dominant_ratio:.2f}).",
                        suggestion="Keep most stops in one main area and move outlier stops to another day.",
                    )
                )
            elif request.pace == Pace.balanced and dominant_ratio < balanced_ratio_floor:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.info,
                        code="area_mix_high",
                        message=f"Day {day.day} mixes several areas (dominant area ratio {dominant_ratio:.2f}).",
                        suggestion="Tighten the route around one stronger area cluster if you want a smoother day.",
                    )
                )

            if cross_area_jump_count >= 2:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.info,
                        code="cross_area_jumps",
                        message=f"Day {day.day} includes {cross_area_jump_count} cross-area jumps.",
                        suggestion="Cluster the day around one district and pick restaurants near the attraction cluster.",
                    )
                )
            if request.pace != Pace.packed and max_leg_km >= 12:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.warning,
                        code="long_route_leg",
                        message=f"Day {day.day} has a long single route leg of about {max_leg_km:.1f} km.",
                        suggestion="Replace the farthest stop with a closer candidate from the main area cluster.",
                    )
                )
            if far_outlier_count >= 1 and request.pace == Pace.relaxed:
                issues.append(
                    CritiqueIssue(
                        severity=IssueSeverity.info,
                        code="far_outlier_stop",
                        message=f"Day {day.day} contains {far_outlier_count} stop(s) far from the main area cluster.",
                        suggestion="Move the outlier stop to another day or swap it for a nearby alternative.",
                    )
                )

    def _check_preferences(
        self,
        request: TripRequest,
        itinerary: Itinerary,
        memory: UserMemory,
        issues: list[CritiqueIssue],
    ) -> None:
        preferences = set(memory.likes) | set(request.preferences)
        if not preferences:
            return
        item_tags = " ".join(item.notes for day in itinerary.days for item in day.items)
        missing = sorted(tag for tag in preferences if tag not in item_tags)
        if missing:
            issues.append(
                CritiqueIssue(
                    severity=IssueSeverity.warning,
                    code="preference_undercovered",
                    message=f"Preferences not clearly covered: {', '.join(missing)}.",
                    suggestion="Replace one generic attraction with a better matching MCP tool result.",
                )
            )


def _dedupe_issues(issues: list[CritiqueIssue]) -> list[CritiqueIssue]:
    result: list[CritiqueIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (issue.code, issue.message)
        if key not in seen:
            result.append(issue)
            seen.add(key)
    return result


def _filter_budget_false_positives(
    request: TripRequest,
    itinerary: Itinerary,
    issues: list[CritiqueIssue],
) -> list[CritiqueIssue]:
    if request.budget is None or itinerary.budget.total > request.budget:
        return issues
    result: list[CritiqueIssue] = []
    for issue in issues:
        text = f"{issue.code} {issue.message}".lower()
        if "budget" in text and ("exceed" in text or "超" in text):
            continue
        result.append(issue)
    return result


def _evaluation_metrics(
    request: TripRequest,
    itinerary: Itinerary,
    issues: list[CritiqueIssue],
) -> dict[str, float]:
    total_items = sum(len(day.items) for day in itinerary.days) or 1
    unique_titles = {item.title for day in itinerary.days for item in day.items}
    preference_tags = set(request.preferences)
    item_notes = " ".join(item.notes for day in itinerary.days for item in day.items)
    covered_preferences = sum(1 for tag in preference_tags if tag in item_notes)
    total_transit_minutes = sum(transit.minutes for day in itinerary.days for transit in day.transit)
    total_scheduled_minutes = sum(item.duration_minutes for day in itinerary.days for item in day.items) + total_transit_minutes
    budget_ratio = itinerary.budget.total / request.budget if request.budget else 0
    dominant_area_ratios = [_dominant_area_ratio(_area_counts(day), len(day.items)) for day in itinerary.days]
    cross_area_jumps = sum(_cross_area_jump_count(day) for day in itinerary.days)
    leg_distances = [transit.distance_km for day in itinerary.days for transit in day.transit if transit.distance_km is not None]
    far_outlier_stop_count = sum(_far_outlier_stop_count(day) for day in itinerary.days)
    return {
        "preference_coverage": round(covered_preferences / len(preference_tags), 3) if preference_tags else 1.0,
        "duplicate_rate": round(1 - len(unique_titles) / total_items, 3),
        "avg_transit_minutes_per_day": round(total_transit_minutes / max(len(itinerary.days), 1), 2),
        "scheduled_minutes_per_day": round(total_scheduled_minutes / max(len(itinerary.days), 1), 2),
        "budget_ratio": round(budget_ratio, 3),
        "dominant_area_ratio": round(sum(dominant_area_ratios) / max(len(dominant_area_ratios), 1), 3),
        "cross_area_jump_count": float(cross_area_jumps),
        "avg_leg_km": round(sum(leg_distances) / max(len(leg_distances), 1), 2) if leg_distances else 0.0,
        "max_leg_km": round(max(leg_distances), 2) if leg_distances else 0.0,
        "far_outlier_stop_count": float(far_outlier_stop_count),
        "issue_count": float(len(issues)),
        "error_count": float(sum(1 for issue in issues if issue.severity == IssueSeverity.error)),
        "warning_count": float(sum(1 for issue in issues if issue.severity == IssueSeverity.warning)),
    }


def _area_counts(day) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in day.items:
        area = item.area or "unknown"
        counts[area] = counts.get(area, 0) + 1
    return counts


def _dominant_area_ratio(area_counts: dict[str, int], item_count: int) -> float:
    if item_count <= 0 or not area_counts:
        return 1.0
    return max(area_counts.values()) / item_count


def _cross_area_jump_count(day) -> int:
    count = 0
    previous_area: str | None = None
    for item in day.items:
        if previous_area is not None and item.area != previous_area:
            count += 1
        previous_area = item.area
    return count


def _normalize_time_of_day(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ")
    aliases = {
        "morning": "Morning",
        "am": "Morning",
        "lunch": "Lunch",
        "noon": "Lunch",
        "afternoon": "Afternoon",
        "late afternoon": "Late afternoon",
        "dinner": "Dinner",
        "evening": "Evening",
        "night": "Evening",
    }
    return aliases.get(normalized, value.strip().title())


def _max_leg_km(day) -> float:
    distances = [transit.distance_km for transit in day.transit if transit.distance_km is not None]
    return max(distances) if distances else 0.0


def _far_outlier_stop_count(day) -> int:
    area_counts = _area_counts(day)
    if not area_counts:
        return 0
    dominant_area = max(area_counts.items(), key=lambda item: item[1])[0]
    count = 0
    for item in day.items:
        if item.area != dominant_area:
            count += 1
    return count


def _critic_system_prompt() -> str:
    return """
你是 TripMind 的 CriticAgent，也是一个专业的旅行行程审查助手。你不负责重新规划整份行程，你的职责是检查现有行程是否靠谱、是否贴合用户需求、是否存在明显风险。

你要做的是“结构化审查”，不是自由聊天，也不是泛泛而谈的反思。

你主要检查以下几类问题：
- 路线压力是否过大：一天是否塞得太满、跨区移动是否过多、节奏是否和用户要求冲突
- 区域一致性是否足够：一天是否围绕少数主区域展开，是否存在明显远离主区域的 outlier stop
- 预算风险是否明显：是否超过用户预算，或者与“预算敏感”的用户画像不匹配
- 偏好是否被覆盖：用户喜欢的内容是否真正出现在行程里，用户明确避开的内容是否被忽略
- 候选地点质量是否可疑：是否像公司、酒店、办公楼、连锁快餐、普通连锁咖啡、或不像旅行者自然会去的点
- 假设是否过多：是否依赖不确定的开放时间、预约条件、日期限制、交通前提
- 是否需要人工确认：某些关键假设如果不确认，后续执行风险较大

下面这些英文词只是返回结果里的字段名，不是给用户看的自然语言：
- `issues`：问题列表
- `metrics`：量化指标
- `passed`：是否通过审查

你的输出结构固定包含 3 个部分：

1. `issues`
这是最重要的部分。每一项都表示一个具体问题。

每个问题对象包含：
- `severity`：严重程度，只能是 `info`、`warning`、`error`
- `code`：稳定、简短的英文代码，使用 snake_case，例如 `budget_exceeded`
- `message`：指出具体问题，尽量说清楚发生在哪一天、哪一类安排上
- `suggestion`：给出可执行的修改建议，避免空泛表达

严重程度建议这样理解：
- `info`：提示类问题，不一定影响可执行性
- `warning`：有明显优化空间，可能影响体验
- `error`：和用户核心约束冲突，或会显著影响方案成立

2. `metrics`
这是量化指标字典。
如果你能从现有输入中稳定判断，可以填写少量指标；如果没有把握，可以返回空对象 `{}`，不要编造数字。
如果你能判断，优先考虑这些指标：
- `dominant_area_ratio`
- `cross_area_jump_count`
- `avg_leg_km`
- `max_leg_km`
- `far_outlier_stop_count`
- `scheduled_minutes_per_day`
- `budget_ratio`

3. `passed`
表示这份行程是否通过审查。
如果存在明显 `error`，通常应为 `false`；否则可以为 `true`。

你会收到三段结构化输入，它们的名字和含义分别是：
- `TripRequest`：本次用户旅行需求，包括目的地、天数、预算、偏好、避雷点、节奏等
- `UserMemory`：用户长期偏好记忆，包括喜欢什么、不喜欢什么、偏好的节奏、是否预算敏感
- `Itinerary`：当前待审查的行程方案，包括每天安排、交通段、预算汇总、补充假设等

关键规则：
- 只根据给你的 `TripRequest`、`UserMemory`、`Itinerary` 这三段输入审查
- 不要编造外部事实，不要编造营业时间、闭馆日、预约规则、真实交通时长
- 不要因为“你猜测可能有问题”就下结论，只有在输入中有足够迹象时才提 issue
- issue 数量宁可少而准，也不要堆很多空泛提醒
- `message` 要指出具体问题，`suggestion` 要能直接指导修改

预算相关特别规则：
- `day.estimated_cost` 不含 buffer
- 整体 `itinerary.budget.total` 包含 buffer
- 除非算术真的冲突，或者整体总预算明显超过用户预算，否则不要仅仅因为两者口径不同就判定预算错误

地点质量相关特别规则：
- 如果某个地点看起来像公司、酒店、办公楼、住宅、普通连锁快餐、普通连锁咖啡，或者不像“值得专门安排行程去”的旅行点，可以提出地点质量风险
- 这类问题更适合写成“候选点不够像旅行目的地 / 不够像本地餐饮体验”，并建议替换

偏好审查相关规则：
- 如果用户说喜欢博物馆、美食、citywalk、在地体验等，你要检查这些偏好是否真的在行程里被体现
- 如果用户说不要太赶、人太多、太贵，也要看行程是否违背这些限制
- 如果只是“可能不够像”，但证据不足，可以降为 `info` 或写得更克制

人工确认相关规则：
- 如果行程依赖关键但未确认的信息，例如具体出发日期、预约要求、闭馆风险、雨天替代方案、预算上限解释等，可以提示需要人工确认
- 但不要把所有不确定性都升级成严重问题

输出风格要求：
- 结论简洁
- 不写长段分析
- 不重复同一个问题
- 严格输出结构化结果

你可以把期望输出理解成这样：
{
  "issues": [
    {
      "severity": "warning",
      "code": "pace_too_full",
      "message": "第 2 天安排较满，跨区域移动较多，和用户希望轻松一些的节奏不完全一致。",
      "suggestion": "删去一个次优景点，或把晚餐改到下午活动区域附近。"
    }
  ],
  "metrics": {},
  "passed": true
}
""".strip()
