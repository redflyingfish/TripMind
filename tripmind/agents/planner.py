from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

from tripmind.llm import OpenAIModelClient
from tripmind.schemas import (
    Attraction,
    DayPlan,
    Itinerary,
    ItineraryDraft,
    ItineraryItem,
    Pace,
    ReplanDirective,
    Restaurant,
    TripRequest,
    UserMemory,
)
from tripmind.travel_data import TRAVEL_PROVIDER, consume_provider_warnings
from tripmind.tools import attractions_search, estimate_budget, estimate_transit, restaurant_search


class PlannerAgent:
    """Build an itinerary from MCP tool results."""

    def __init__(self, llm: OpenAIModelClient | None = None) -> None:
        self.llm = llm

    def plan(
        self,
        request: TripRequest,
        memory: UserMemory,
        directive: ReplanDirective | None = None,
        budget_override: float | None = None,
    ) -> Itinerary:
        preferences = _unique([*memory.likes, *request.preferences])
        avoid = _unique([*memory.dislikes, *request.avoid])
        effective_request = request.model_copy(deep=True)
        if budget_override is not None:
            effective_request.budget = budget_override
        if directive and directive.reduce_intensity and effective_request.pace == Pace.packed:
            effective_request.pace = Pace.balanced
        slots_per_day = _target_activity_count(effective_request.pace)
        if directive and directive.max_attractions_per_day is not None:
            slots_per_day = max(2, min(slots_per_day, directive.max_attractions_per_day))
        consume_provider_warnings()
        attractions, restaurants = self._load_candidates(effective_request, preferences, avoid, slots_per_day)
        attractions = _cluster_attractions(attractions, preferences, effective_request.pace, effective_request.days, slots_per_day)
        restaurants = _cluster_restaurants(restaurants, preferences, effective_request.pace, effective_request.days)
        if directive:
            attractions = _apply_directive_to_attractions(attractions, directive, preferences)
            restaurants = _apply_directive_to_restaurants(restaurants, directive, preferences)

        if self.llm:
            draft = self._draft_with_llm(effective_request, memory, attractions, restaurants, slots_per_day, directive)
            return self._build_from_draft(effective_request, draft, attractions, restaurants, slots_per_day, directive)

        return self._deterministic_plan(effective_request, memory, attractions, restaurants, slots_per_day, directive)

    def plan_variants(self, request: TripRequest, memory: UserMemory) -> list[tuple[str, str, Itinerary]]:
        variants: list[tuple[str, str, Itinerary]] = []
        tiers = _budget_variants(request, memory)
        if not tiers:
            itinerary = self.plan(request, memory)
            return [("default", "标准版", itinerary)]
        for variant_id, label, budget in tiers:
            directive = _budget_variant_directive(variant_id, request)
            itinerary = self.plan(request, memory, directive=directive, budget_override=budget)
            variants.append((variant_id, label, itinerary))
        return _normalize_budget_variant_labels(variants)

    def _load_candidates(
        self,
        request: TripRequest,
        preferences: list[str],
        avoid: list[str],
        slots_per_day: int,
    ) -> tuple[list[Attraction], list[Restaurant]]:
        attraction_kwargs = {
            "destination": request.destination,
            "preferences": preferences,
            "avoid": avoid,
            "max_results": max(request.days * slots_per_day * 3, 12),
        }
        restaurant_kwargs = {
            "destination": request.destination,
            "preferences": preferences,
            "avoid": avoid,
            "max_results": max(request.days * 8, 12),
        }

        if TRAVEL_PROVIDER == "baidu":
            attractions = attractions_search(**attraction_kwargs)
            restaurants = restaurant_search(**restaurant_kwargs)
            return attractions, restaurants

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                attraction_future = executor.submit(attractions_search, **attraction_kwargs)
                restaurant_future = executor.submit(restaurant_search, **restaurant_kwargs)
                return attraction_future.result(), restaurant_future.result()
        except Exception:
            # MCP stdio tool calls can fail transiently under concurrent startup.
            # Retry once in sequence before surfacing the error to the API layer.
            attractions = attractions_search(**attraction_kwargs)
            restaurants = restaurant_search(**restaurant_kwargs)
            return attractions, restaurants

    def _draft_with_llm(
        self,
        request: TripRequest,
        memory: UserMemory,
        attractions: list[Attraction],
        restaurants: list[Restaurant],
        slots_per_day: int,
        directive: ReplanDirective | None = None,
    ) -> ItineraryDraft:
        system = _planner_system_prompt()
        area_clusters = _build_area_cluster_summary(attractions, restaurants, request.preferences, request.pace)
        directive_text = _directive_prompt_fragment(directive)
        user = (
            f"TripRequest:\n{request.model_dump_json(exclude={'raw_text'}, exclude_none=True)}\n\n"
            f"UserMemory:\n{memory.model_dump_json(exclude={'updated_at'}, exclude_none=True)}\n\n"
            f"每天目标景点数: {slots_per_day}\n"
            f"用户指定时段 requested_time_slots: {request.requested_time_slots or '未指定'}\n"
            f"用户避开的时段 avoid_time_slots: {request.avoid_time_slots or '未指定'}\n"
            "每天通常按 Morning 景点、Lunch 餐厅、Afternoon 景点、Dinner 餐厅、Evening 景点组织；"
            "Breakfast 默认不安排，只有用户明确要求早餐时才安排。"
            "如果 pace=relaxed，可以不安排 Evening，必要时也可以从 Lunch 后开始，优先减少跨区移动；"
            "如果用户只要求午餐/晚餐/晚上等局部内容，只安排对应部分。"
            "硬性语义规则：Breakfast/Lunch/Dinner 必须是 restaurant；Morning/Afternoon/Late afternoon/Evening 必须是 attraction。"
            "宁可选距离更近但标签略弱的点，也不要为了标签匹配制造折返。"
            "如果候选餐厅里没有合适正餐，选择最像本地餐厅的候选，不要选择连锁咖啡/快餐。\n\n"
            f"{directive_text}\n\n"
            f"候选区域簇摘要:\n{area_clusters}\n\n"
            f"MCP 候选景点 attractions_search:\n{_dump_attractions(attractions)}\n\n"
            f"MCP 候选餐厅 restaurant_search:\n{_dump_restaurants(restaurants)}\n\n"
            "请返回 ItineraryDraft。stops[].name 必须完全等于候选列表中的 name。"
        )
        return self.llm.parse(system=system, user=user, schema=ItineraryDraft)

    def _deterministic_plan(
        self,
        request: TripRequest,
        memory: UserMemory,
        attractions: list[Attraction],
        restaurants: list[Restaurant],
        slots_per_day: int,
        directive: ReplanDirective | None = None,
    ) -> Itinerary:
        preferences = _unique([*memory.likes, *request.preferences])
        attraction_iter = cycle(attractions)
        restaurant_iter = cycle(restaurants)
        days: list[DayPlan] = []
        attraction_costs: list[float] = []
        restaurant_costs: list[float] = []
        transit_costs: list[float] = []
        used_titles: set[str] = set()
        day_slots = _day_slots_for_request(request, slots_per_day)

        for day_number in range(1, request.days + 1):
            items: list[ItineraryItem] = []
            for slot in day_slots:
                if _expected_kind_for_time(slot) == "restaurant":
                    items.append(_restaurant_item(slot, next(restaurant_iter)))
                else:
                    items.append(
                        _attraction_item(
                            slot,
                            next(attraction_iter),
                            duration_cap=90 if slot == "Evening" else None,
                        )
                    )

            items = _repair_slot_semantics(
                items,
                attraction_iter,
                restaurant_iter,
                day_slots=day_slots,
            )
            items = _compact_day_items(
                items,
                attractions,
                restaurants,
                request.preferences,
                request.pace,
                max_jump_km_override=directive.max_leg_km if directive else None,
            )
            items = _trim_to_requested_slots(items, request, slots_per_day)
            items = _dedupe_day_items(
                items,
                used_titles,
                attractions,
                restaurants,
                request.preferences,
                request.pace,
                max_jump_km_override=directive.max_leg_km if directive else None,
            )

            transits = []
            for previous, current in zip(items, items[1:]):
                transit = estimate_transit(
                    previous.title,
                    current.title,
                    request.pace,
                    origin_latitude=previous.latitude,
                    origin_longitude=previous.longitude,
                    destination_latitude=current.latitude,
                    destination_longitude=current.longitude,
                )
                transits.append(transit)
                transit_costs.append(transit.cost)

            for item in items:
                if item.kind == "attraction":
                    attraction_costs.append(item.estimated_cost)
                elif item.kind == "restaurant":
                    restaurant_costs.append(item.estimated_cost)

            item_cost = sum(item.estimated_cost for item in items)
            transit_cost = sum(transit.cost for transit in transits)
            days.append(
                DayPlan(
                    day=day_number,
                    theme=_theme_for_items(items, preferences),
                    items=items,
                    transit=transits,
                    estimated_cost=round((item_cost + transit_cost) * request.travelers, 2),
                )
            )

        budget = estimate_budget(
            attraction_costs=attraction_costs,
            restaurant_costs=restaurant_costs,
            transit_costs=transit_costs,
            travelers=request.travelers,
            currency=request.currency,
        )

        assumptions = [
            "Uses MCP travel tools for attraction, restaurant, transit, and budget data.",
            "Hotel, flight, and long-distance transport are outside this first demo scope.",
        ]
        if request.start_date is None:
            assumptions.append("No start date was provided; weather and opening-day constraints are not checked.")

        return Itinerary(request=request, days=days, budget=budget, assumptions=assumptions)

    def _build_from_draft(
        self,
        request: TripRequest,
        draft: ItineraryDraft,
        attractions: list[Attraction],
        restaurants: list[Restaurant],
        slots_per_day: int,
        directive: ReplanDirective | None = None,
    ) -> Itinerary:
        attraction_by_name = {item.name: item for item in attractions}
        restaurant_by_name = {item.name: item for item in restaurants}
        fallback_attractions = cycle(attractions)
        fallback_restaurants = cycle(restaurants)
        days: list[DayPlan] = []
        attraction_costs: list[float] = []
        restaurant_costs: list[float] = []
        transit_costs: list[float] = []
        used_titles: set[str] = set()
        day_slots = _day_slots_for_request(request, slots_per_day)

        for day_number in range(1, request.days + 1):
            draft_day = next((day for day in draft.days if day.day == day_number), None)
            stops = draft_day.stops if draft_day else []
            items: list[ItineraryItem] = []
            attraction_count = 0

            for stop in stops:
                normalized_time = _normalize_time_of_day(stop.time_of_day)
                expected_kind = _expected_kind_for_time(normalized_time)
                if (
                    expected_kind == "attraction"
                    and stop.name in attraction_by_name
                    and attraction_count < slots_per_day
                ):
                    attraction = attraction_by_name[stop.name]
                    items.append(_attraction_item(normalized_time, attraction, notes_suffix=stop.rationale))
                    attraction_count += 1
                elif expected_kind == "restaurant" and stop.name in restaurant_by_name:
                    restaurant = restaurant_by_name[stop.name]
                    items.append(_restaurant_item(normalized_time, restaurant, notes_suffix=stop.rationale))
                elif stop.kind == "attraction" and stop.name in attraction_by_name:
                    if attraction_count >= slots_per_day:
                        continue
                    attraction = attraction_by_name[stop.name]
                    items.append(_attraction_item(_next_activity_slot(items), attraction, notes_suffix=stop.rationale))
                    attraction_count += 1
                elif stop.kind == "restaurant" and stop.name in restaurant_by_name:
                    restaurant = restaurant_by_name[stop.name]
                    items.append(_restaurant_item(_next_meal_slot(items), restaurant, notes_suffix=stop.rationale))

            items = _fill_day_items(items, fallback_attractions, fallback_restaurants, day_slots)
            items = _repair_slot_semantics(
                items,
                fallback_attractions,
                fallback_restaurants,
                day_slots=day_slots,
            )
            items = _compact_day_items(
                items,
                attractions,
                restaurants,
                request.preferences,
                request.pace,
                max_jump_km_override=directive.max_leg_km if directive else None,
            )
            items = _trim_to_requested_slots(items, request, slots_per_day)
            items = _dedupe_day_items(
                items,
                used_titles,
                attractions,
                restaurants,
                request.preferences,
                request.pace,
                max_jump_km_override=directive.max_leg_km if directive else None,
            )
            transits = []
            for previous, current in zip(items, items[1:]):
                transit = estimate_transit(
                    previous.title,
                    current.title,
                    request.pace,
                    origin_latitude=previous.latitude,
                    origin_longitude=previous.longitude,
                    destination_latitude=current.latitude,
                    destination_longitude=current.longitude,
                )
                transits.append(transit)
                transit_costs.append(transit.cost)

            for item in items:
                if item.kind == "attraction":
                    attraction_costs.append(item.estimated_cost)
                elif item.kind == "restaurant":
                    restaurant_costs.append(item.estimated_cost)

            item_cost = sum(item.estimated_cost for item in items)
            transit_cost = sum(transit.cost for transit in transits)
            days.append(
                DayPlan(
                    day=day_number,
                    theme=draft_day.theme if draft_day and draft_day.theme.strip() else _theme_for_items(items, request.preferences),
                    items=items,
                    transit=transits,
                    estimated_cost=round((item_cost + transit_cost) * request.travelers, 2),
                )
            )

        budget = estimate_budget(
            attraction_costs=attraction_costs,
            restaurant_costs=restaurant_costs,
            transit_costs=transit_costs,
            travelers=request.travelers,
            currency=request.currency,
        )
        assumptions = [
            "LLM selected from MCP tool candidates; transit and budget were calculated by MCP tools.",
            "Hotel, flight, and long-distance transport are outside this first demo scope.",
        ]
        if request.start_date is None:
            assumptions.append("No start date was provided; weather and opening-day constraints are not checked.")
        assumptions.extend(draft.assumptions)

        return Itinerary(
            request=request,
            days=days,
            budget=budget,
            assumptions=_unique(assumptions),
            adjustment_suggestions=_unique(draft.adjustment_suggestions),
        )


def _attraction_item(time_of_day: str, attraction, duration_cap: int | None = None, notes_suffix: str = "") -> ItineraryItem:
    duration = min(attraction.duration_minutes, duration_cap) if duration_cap else attraction.duration_minutes
    notes = ", ".join(attraction.tags)
    if notes_suffix:
        notes = f"{notes}; rationale: {notes_suffix}"
    return ItineraryItem(
        time_of_day=time_of_day,
        title=attraction.name,
        kind="attraction",
        area=attraction.area,
        duration_minutes=duration,
        estimated_cost=attraction.ticket_price,
        notes=notes,
        source=attraction.source,
        provider_note=attraction.provider_note,
        rating=attraction.rating,
        review_count=attraction.review_count,
        latitude=attraction.latitude,
        longitude=attraction.longitude,
    )


def _restaurant_item(time_of_day: str, restaurant, notes_suffix: str = "") -> ItineraryItem:
    notes = ", ".join(restaurant.tags)
    if notes_suffix:
        notes = f"{notes}; rationale: {notes_suffix}"
    return ItineraryItem(
        time_of_day=time_of_day,
        title=restaurant.name,
        kind="restaurant",
        area=restaurant.area,
        duration_minutes=70,
        estimated_cost=restaurant.average_price,
        notes=notes,
        source=restaurant.source,
        provider_note=restaurant.provider_note,
        rating=restaurant.rating,
        review_count=restaurant.review_count,
        latitude=restaurant.latitude,
        longitude=restaurant.longitude,
    )


def _theme_for_items(items: list[ItineraryItem], preferences: list[str]) -> str:
    attraction_tags = [
        tag
        for item in items
        if item.kind == "attraction"
        for tag in _tags_from_notes(item.notes)
    ]
    area_counts: dict[str, int] = {}
    for item in items:
        area_counts[item.area] = area_counts.get(item.area, 0) + 1
    anchor_area = max(area_counts.items(), key=lambda entry: entry[1])[0] if area_counts else ""
    primary_tag = _primary_theme_tag(attraction_tags, preferences)
    if anchor_area and primary_tag:
        return f"{anchor_area}{primary_tag}"
    if primary_tag:
        return primary_tag
    if anchor_area:
        return f"{anchor_area}漫游线"
    return "Balanced local highlights"


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _dump_attractions(attractions: list[Attraction]) -> str:
    rows = [
        {
            "name": item.name,
            "area": item.area,
            "tags": item.tags,
            "minutes": item.duration_minutes,
            "price": item.ticket_price,
            "crowd": item.crowd_level,
            "source": item.source,
            "rating": item.rating,
            "review_count": item.review_count,
        }
        for item in attractions
    ]
    return json_compact(rows)


def _dump_restaurants(restaurants: list[Restaurant]) -> str:
    rows = [
        {
            "name": item.name,
            "area": item.area,
            "tags": item.tags,
            "price": item.average_price,
            "source": item.source,
            "rating": item.rating,
            "review_count": item.review_count,
        }
        for item in restaurants
    ]
    return json_compact(rows)


def _build_area_cluster_summary(
    attractions: list[Attraction],
    restaurants: list[Restaurant],
    preferences: list[str],
    pace: Pace,
) -> str:
    area_info: dict[str, dict[str, object]] = {}
    for attraction in attractions:
        info = area_info.setdefault(attraction.area, {"attractions": [], "restaurants": []})
        info["attractions"].append(attraction)
    for restaurant in restaurants:
        info = area_info.setdefault(restaurant.area, {"attractions": [], "restaurants": []})
        info["restaurants"].append(restaurant)

    ranked_areas = sorted(
        area_info.items(),
        key=lambda entry: _area_cluster_score(entry[1]["attractions"], entry[1]["restaurants"], preferences, pace),
        reverse=True,
    )

    lines: list[str] = []
    for area, info in ranked_areas[:4]:
        top_attractions = ", ".join(item.name for item in info["attractions"][:2]) or "无明显景点"
        top_restaurants = ", ".join(item.name for item in info["restaurants"][:2]) or "无明显餐厅"
        lines.append(
            f"- {area}: 景点 {len(info['attractions'])} 个，餐厅 {len(info['restaurants'])} 个；"
            f"代表景点：{top_attractions}；代表餐厅：{top_restaurants}"
        )
    return "\n".join(lines) or "- 暂无明显区域簇"


def json_compact(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _directive_prompt_fragment(directive: ReplanDirective | None) -> str:
    if directive is None:
        return "本轮为首次规划。"
    notes = "；".join(directive.notes) or "无额外说明"
    blocked = ", ".join(directive.blocked_titles) or "无"
    preserve = ", ".join(directive.preserve_titles) or "无"
    preferred_areas = ", ".join(directive.preferred_areas) or "无"
    avoid_areas = ", ".join(directive.avoid_areas) or "无"
    required_preferences = ", ".join(directive.required_preferences) or "无"
    return (
        "这是一次受约束重规划，请优先修复上轮问题。\n"
        f"- 说明: {notes}\n"
        f"- 不要再选这些点: {blocked}\n"
        f"- 尽量保留这些点: {preserve}\n"
        f"- 优先区域: {preferred_areas}\n"
        f"- 尽量回避这些区域: {avoid_areas}\n"
        f"- 必须更好覆盖的偏好: {required_preferences}\n"
        f"- 是否更偏省钱: {'是' if directive.prefer_cheaper else '否'}\n"
        f"- 是否更强调区域集中: {'是' if directive.tighter_area_clustering else '否'}\n"
        f"- 是否降低强度: {'是' if directive.reduce_intensity else '否'}\n"
        f"- 每天最多景点数: {directive.max_attractions_per_day if directive.max_attractions_per_day is not None else '保持默认'}\n"
        f"- 目标单段路程上限: {directive.max_leg_km if directive.max_leg_km is not None else '保持默认'} km"
    )


def _fill_day_items(
    items: list[ItineraryItem],
    attractions,
    restaurants,
    day_slots: list[str],
) -> list[ItineraryItem]:
    ordered = sorted(items, key=lambda item: _time_order(item.time_of_day))
    present_slots = {_normalize_time_of_day(item.time_of_day) for item in ordered}
    for slot in day_slots:
        if slot in present_slots:
            continue
        if _expected_kind_for_time(slot) == "restaurant":
            ordered.append(_restaurant_item(slot, next(restaurants)))
        else:
            ordered.append(
                _attraction_item(
                    slot,
                    next(attractions),
                    duration_cap=90 if slot == "Evening" else None,
                )
            )

    return sorted(ordered, key=lambda item: _time_order(item.time_of_day))


def _repair_slot_semantics(
    items: list[ItineraryItem],
    attractions,
    restaurants,
    day_slots: list[str],
) -> list[ItineraryItem]:
    """Ensure meal slots contain restaurants and activity slots contain attractions."""
    repaired: list[ItineraryItem] = []
    used_titles = {item.title for item in items}
    activity_slots = [slot for slot in day_slots if _expected_kind_for_time(slot) == "attraction"]
    meal_slots = [slot for slot in day_slots if _expected_kind_for_time(slot) == "restaurant"]

    for item in sorted(items, key=lambda item: _time_order(item.time_of_day)):
        time_of_day = _normalize_time_of_day(item.time_of_day)
        expected_kind = _expected_kind_for_time(time_of_day)
        if expected_kind == item.kind:
            item.time_of_day = time_of_day
            repaired.append(item)
            continue

        if expected_kind == "restaurant":
            replacement = _next_unique_restaurant_item(time_of_day, restaurants, used_titles)
        elif expected_kind == "attraction":
            replacement = _next_unique_attraction_item(
                time_of_day,
                attractions,
                used_titles,
                duration_cap=90 if time_of_day == "Evening" else None,
            )
        else:
            replacement = item
            replacement.time_of_day = time_of_day
        used_titles.add(replacement.title)
        repaired.append(replacement)

    present_meals = {_normalize_time_of_day(item.time_of_day) for item in repaired if item.kind == "restaurant"}
    for slot in meal_slots:
        if slot not in present_meals:
            repaired.append(_next_unique_restaurant_item(slot, restaurants, used_titles))

    present_activity_slots = [
        _normalize_time_of_day(item.time_of_day)
        for item in repaired
        if item.kind == "attraction"
    ]
    for slot in activity_slots:
        if slot in present_activity_slots:
            continue
        repaired.append(
            _next_unique_attraction_item(
                slot,
                attractions,
                used_titles,
                duration_cap=90 if slot == "Evening" else None,
            )
        )
        present_activity_slots.append(slot)

    return sorted(repaired, key=lambda item: _time_order(item.time_of_day))


def _next_unique_restaurant_item(time_of_day: str, restaurants, used_titles: set[str]) -> ItineraryItem:
    for _ in range(200):
        restaurant = next(restaurants)
        if restaurant.name in used_titles:
            continue
        used_titles.add(restaurant.name)
        return _restaurant_item(time_of_day, restaurant, notes_suffix="repaired meal slot")
    restaurant = next(restaurants)
    return _restaurant_item(time_of_day, restaurant, notes_suffix="repaired meal slot")


def _next_unique_attraction_item(
    time_of_day: str,
    attractions,
    used_titles: set[str],
    duration_cap: int | None = None,
) -> ItineraryItem:
    for _ in range(200):
        attraction = next(attractions)
        if attraction.name in used_titles:
            continue
        used_titles.add(attraction.name)
        return _attraction_item(
            time_of_day,
            attraction,
            duration_cap=duration_cap,
            notes_suffix="repaired activity slot",
        )
    attraction = next(attractions)
    return _attraction_item(
        time_of_day,
        attraction,
        duration_cap=duration_cap,
        notes_suffix="repaired activity slot",
    )


def _target_activity_count(pace: Pace) -> int:
    return {Pace.relaxed: 2, Pace.balanced: 3, Pace.packed: 4}[pace]


def _activity_slots_for_plan(slots_per_day: int, pace: Pace) -> list[str]:
    if pace == Pace.relaxed:
        return ["Morning", "Afternoon"][:slots_per_day]
    if slots_per_day <= 2:
        return ["Morning", "Afternoon"][:slots_per_day]
    if slots_per_day == 3:
        return ["Morning", "Afternoon", "Evening"]
    return ["Morning", "Afternoon", "Late afternoon", "Evening"][:slots_per_day]


def _day_slots_for_request(request: TripRequest, slots_per_day: int) -> list[str]:
    if request.requested_time_slots:
        requested = [_normalize_time_of_day(slot) for slot in request.requested_time_slots]
        requested = [slot for slot in requested if _normalize_time_of_day(slot) not in {_normalize_time_of_day(x) for x in request.avoid_time_slots}]
        return requested or ["Afternoon", "Dinner"]

    default_slots = _default_day_slots(slots_per_day, request.pace)
    avoid = {_normalize_time_of_day(slot) for slot in request.avoid_time_slots}
    filtered = [slot for slot in default_slots if slot not in avoid]
    return filtered or ["Afternoon", "Dinner"]


def _default_day_slots(slots_per_day: int, pace: Pace) -> list[str]:
    activity_slots = _activity_slots_for_plan(slots_per_day, pace)
    slots: list[str] = []
    if "Morning" in activity_slots:
        slots.append("Morning")
    slots.append("Lunch")
    if "Afternoon" in activity_slots:
        slots.append("Afternoon")
    if "Late afternoon" in activity_slots:
        slots.append("Late afternoon")
    slots.append("Dinner")
    if "Evening" in activity_slots:
        slots.append("Evening")
    return slots


def _trim_to_requested_slots(items: list[ItineraryItem], request: TripRequest, slots_per_day: int) -> list[ItineraryItem]:
    allowed = set(_day_slots_for_request(request, slots_per_day))
    return [
        item
        for item in sorted(items, key=lambda item: _time_order(item.time_of_day))
        if _normalize_time_of_day(item.time_of_day) in allowed
    ]


def _expected_kind_for_time(time_of_day: str) -> str | None:
    normalized = _normalize_time_of_day(time_of_day)
    if normalized in {"Breakfast", "Lunch", "Dinner"}:
        return "restaurant"
    if normalized in {"Morning", "Afternoon", "Late afternoon", "Evening"}:
        return "attraction"
    return None


def _next_activity_slot(items: list[ItineraryItem]) -> str:
    used = {_normalize_time_of_day(item.time_of_day) for item in items if item.kind == "attraction"}
    for slot in ["Morning", "Afternoon", "Late afternoon", "Evening"]:
        if slot not in used:
            return slot
    return "Evening"


def _next_meal_slot(items: list[ItineraryItem]) -> str:
    used = {_normalize_time_of_day(item.time_of_day) for item in items if item.kind == "restaurant"}
    if "Breakfast" not in used:
        return "Breakfast"
    if "Lunch" not in used:
        return "Lunch"
    return "Dinner"


def _budget_variant_directive(variant_id: str, request: TripRequest) -> ReplanDirective | None:
    if variant_id == "economy":
        return ReplanDirective(
            label="economy",
            notes=["Prefer cheaper attractions and restaurants for the economy variant."],
            prefer_cheaper=True,
            reduce_intensity=True,
            tighter_area_clustering=True,
            max_attractions_per_day=2 if request.days >= 1 else None,
            max_leg_km=8.0,
        )
    if variant_id == "comfort":
        return ReplanDirective(
            label="comfort",
            notes=["Comfort variant can keep higher-quality paid options if the route stays coherent."],
            prefer_cheaper=False,
            reduce_intensity=False,
        )
    return None


def _normalize_budget_variant_labels(variants: list[tuple[str, str, Itinerary]]) -> list[tuple[str, str, Itinerary]]:
    if len(variants) <= 1:
        return variants
    ordered = sorted(variants, key=lambda item: item[2].budget.total)
    canonical = [("economy", "省钱版"), ("standard", "均衡版"), ("comfort", "体验版")]
    normalized: list[tuple[str, str, Itinerary]] = []
    for (variant_id, label), (_, _, itinerary) in zip(canonical, ordered):
        normalized.append((variant_id, label, itinerary))
    return normalized


def _dedupe_day_items(
    items: list[ItineraryItem],
    used_titles: set[str],
    attractions: list[Attraction],
    restaurants: list[Restaurant],
    preferences: list[str],
    pace: Pace,
    max_jump_km_override: float | None = None,
) -> list[ItineraryItem]:
    ordered = sorted(items, key=lambda item: _time_order(item.time_of_day))
    local_seen: set[str] = set()
    max_jump_km = max_jump_km_override or {Pace.relaxed: 8.0, Pace.balanced: 12.0, Pace.packed: 16.0}[pace]

    for index, item in enumerate(ordered):
        if item.title not in local_seen and item.title not in used_titles:
            local_seen.add(item.title)
            continue

        replacement = _replacement_for_duplicate(
            index=index,
            items=ordered,
            disallowed_titles=used_titles | local_seen | {other.title for pos, other in enumerate(ordered) if pos != index},
            attractions=attractions,
            restaurants=restaurants,
            preferences=preferences,
            max_jump_km=max_jump_km,
        )
        if replacement is not None:
            ordered[index] = replacement
        local_seen.add(ordered[index].title)

    used_titles.update(item.title for item in ordered)
    return ordered


def _replacement_for_duplicate(
    index: int,
    items: list[ItineraryItem],
    disallowed_titles: set[str],
    attractions: list[Attraction],
    restaurants: list[Restaurant],
    preferences: list[str],
    max_jump_km: float,
) -> ItineraryItem | None:
    current = items[index]
    previous = items[index - 1] if index > 0 else None
    next_item = items[index + 1] if index + 1 < len(items) else None

    if current.kind == "attraction":
        candidates = [
            _attraction_item(
                current.time_of_day,
                item,
                duration_cap=current.duration_minutes if current.time_of_day == "Evening" else None,
                notes_suffix="reranked to avoid duplicates",
            )
            for item in attractions
            if item.name not in disallowed_titles
        ]
    else:
        candidates = [
            _restaurant_item(current.time_of_day, item, notes_suffix="reranked to avoid duplicates")
            for item in restaurants
            if item.name not in disallowed_titles
        ]

    ranked = sorted(
        candidates,
        key=lambda candidate: _duplicate_replacement_score(previous, candidate, next_item, preferences, max_jump_km),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _duplicate_replacement_score(
    previous: ItineraryItem | None,
    candidate: ItineraryItem,
    next_item: ItineraryItem | None,
    preferences: list[str],
    max_jump_km: float,
) -> float:
    score = _route_quality_signal(candidate.rating, candidate.review_count)
    score += sum(2.5 for tag in _tags_from_notes(candidate.notes) if tag in preferences)
    score += 0.8 if candidate.source in {"baidu_place", "amap_place"} else 0.0
    if previous is not None:
        score += _pair_route_fit_score(previous, candidate, max_jump_km)
    if next_item is not None:
        score += _pair_route_fit_score(candidate, next_item, max_jump_km)
    return score


def _pair_route_fit_score(first: ItineraryItem, second: ItineraryItem, max_jump_km: float) -> float:
    distance = _item_distance_km(first, second)
    if distance is None:
        return 0.0
    area_bonus = 1.8 if first.area == second.area else 0.0
    distance_penalty = min(distance, max_jump_km * 2) * 4.5
    return area_bonus - distance_penalty


def _apply_directive_to_attractions(
    attractions: list[Attraction],
    directive: ReplanDirective,
    preferences: list[str],
) -> list[Attraction]:
    return _rerank_with_directive(
        attractions,
        blocked_titles=set(directive.blocked_titles),
        preserve_titles=set(directive.preserve_titles),
        preferred_areas=set(directive.preferred_areas),
        avoid_areas=set(directive.avoid_areas),
        required_preferences=set(directive.required_preferences or preferences),
        prefer_cheaper=directive.prefer_cheaper,
        tighter_area_clustering=directive.tighter_area_clustering,
        is_restaurant=False,
    )


def _apply_directive_to_restaurants(
    restaurants: list[Restaurant],
    directive: ReplanDirective,
    preferences: list[str],
) -> list[Restaurant]:
    return _rerank_with_directive(
        restaurants,
        blocked_titles=set(directive.blocked_titles),
        preserve_titles=set(directive.preserve_titles),
        preferred_areas=set(directive.preferred_areas),
        avoid_areas=set(directive.avoid_areas),
        required_preferences=set(directive.required_preferences or preferences),
        prefer_cheaper=directive.prefer_cheaper,
        tighter_area_clustering=directive.tighter_area_clustering,
        is_restaurant=True,
    )


def _rerank_with_directive(
    items,
    blocked_titles: set[str],
    preserve_titles: set[str],
    preferred_areas: set[str],
    avoid_areas: set[str],
    required_preferences: set[str],
    prefer_cheaper: bool,
    tighter_area_clustering: bool,
    is_restaurant: bool,
):
    filtered = [
        item
        for item in items
        if getattr(item, "name", None) not in blocked_titles
        and getattr(item, "area", "") not in avoid_areas
    ]
    if not filtered:
        filtered = [item for item in items if getattr(item, "name", None) not in blocked_titles] or items

    def score(item) -> tuple[float, float, float]:
        area_bonus = 2.5 if preferred_areas and getattr(item, "area", "") in preferred_areas else 0.0
        preserve_bonus = 3.5 if getattr(item, "name", None) in preserve_titles else 0.0
        pref_bonus = sum(2.0 for tag in getattr(item, "tags", []) if tag in required_preferences)
        cost = getattr(item, "average_price", None) if is_restaurant else getattr(item, "ticket_price", 0.0)
        cost_bonus = -min(cost / (80 if is_restaurant else 60), 3.0) if prefer_cheaper else 0.0
        cluster_bonus = area_bonus if tighter_area_clustering else 0.0
        quality = _route_quality_signal(getattr(item, "rating", None), getattr(item, "review_count", None))
        return (quality + pref_bonus + cluster_bonus + cost_bonus + preserve_bonus, preserve_bonus + area_bonus, pref_bonus)

    return sorted(filtered, key=score, reverse=True)


def _budget_variants(request: TripRequest, memory: UserMemory) -> list[tuple[str, str, float]]:
    if request.budget is not None:
        return []
    base = max(350 * request.days * request.travelers, 350)
    if memory.budget_sensitive:
        return [
            ("economy", "省钱版", round(base * 0.8, 0)),
            ("standard", "均衡版", round(base, 0)),
        ]
    return [
        ("economy", "省钱版", round(base * 0.85, 0)),
        ("standard", "均衡版", round(base, 0)),
        ("comfort", "体验版", round(base * 1.35, 0)),
    ]


def _compact_day_items(
    items: list[ItineraryItem],
    attractions: list[Attraction],
    restaurants: list[Restaurant],
    preferences: list[str],
    pace: Pace,
    max_jump_km_override: float | None = None,
) -> list[ItineraryItem]:
    if len(items) < 2:
        return sorted(items, key=lambda item: _time_order(item.time_of_day))

    max_jump_km = max_jump_km_override or {Pace.relaxed: 8.0, Pace.balanced: 12.0, Pace.packed: 16.0}[pace]
    attraction_by_name = {item.name: item for item in attractions}
    restaurant_by_name = {item.name: item for item in restaurants}
    ordered = sorted(items, key=lambda item: _time_order(item.time_of_day))
    used_titles = {item.title for item in ordered}

    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        distance = _item_distance_km(previous, current)
        if distance is None or distance <= max_jump_km:
            continue

        replacement = _find_nearer_candidate(
            previous=previous,
            current=current,
            used_titles=used_titles,
            attraction_by_name=attraction_by_name,
            restaurant_by_name=restaurant_by_name,
            preferences=preferences,
            max_jump_km=max_jump_km,
        )
        if replacement is None:
            continue

        used_titles.discard(current.title)
        used_titles.add(replacement.title)
        ordered[index] = replacement

    return ordered


def _find_nearer_candidate(
    previous: ItineraryItem,
    current: ItineraryItem,
    used_titles: set[str],
    attraction_by_name: dict[str, Attraction],
    restaurant_by_name: dict[str, Restaurant],
    preferences: list[str],
    max_jump_km: float,
) -> ItineraryItem | None:
    if current.kind == "attraction":
        candidates = [
            item
            for item in attraction_by_name.values()
            if item.name not in used_titles and item.latitude is not None and item.longitude is not None
        ]
        items = [
            _attraction_item(
                current.time_of_day,
                item,
                duration_cap=current.duration_minutes if current.time_of_day == "Evening" else None,
                notes_suffix="reranked for route compactness",
            )
            for item in candidates
        ]
    else:
        candidates = [
            item
            for item in restaurant_by_name.values()
            if item.name not in used_titles and item.latitude is not None and item.longitude is not None
        ]
        items = [
            _restaurant_item(current.time_of_day, item, notes_suffix="reranked for route compactness")
            for item in candidates
        ]

    ranked = sorted(
        items,
        key=lambda item: (
            _candidate_route_score(previous, item, preferences, max_jump_km),
            item.rating or 0,
            item.review_count or 0,
            -item.estimated_cost if current.kind == "restaurant" else 0,
        ),
        reverse=True,
    )
    if not ranked:
        return None

    best = ranked[0]
    if _item_distance_km(previous, best) is None or _item_distance_km(previous, best) > max_jump_km * 1.15:
        return None
    return best


def _candidate_route_score(
    previous: ItineraryItem,
    candidate: ItineraryItem,
    preferences: list[str],
    max_jump_km: float,
) -> float:
    distance = _item_distance_km(previous, candidate)
    if distance is None:
        return -999
    preference_bonus = sum(3 for tag in _tags_from_notes(candidate.notes) if tag in preferences)
    source_bonus = 1 if candidate.source in {"baidu_place", "amap_place"} else 0
    quality_bonus = _route_quality_signal(candidate.rating, candidate.review_count)
    distance_penalty = min(distance, max_jump_km * 2) * 5
    return preference_bonus + source_bonus + quality_bonus - distance_penalty


def _tags_from_notes(notes: str) -> list[str]:
    leading = notes.split(";", 1)[0]
    return [part.strip() for part in leading.split(",") if part.strip()]


def _item_distance_km(first: ItineraryItem, second: ItineraryItem) -> float | None:
    if first.latitude is None or first.longitude is None or second.latitude is None or second.longitude is None:
        return None
    return _haversine_km(first.latitude, first.longitude, second.latitude, second.longitude)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _route_quality_signal(rating: float | None, review_count: int | None) -> float:
    if not rating or rating <= 0:
        return 0.0
    count = max(review_count or 0, 0)
    prior_mean = 4.2
    prior_weight = 80
    weighted_rating = ((prior_weight * prior_mean) + (count * rating)) / (prior_weight + count)
    popularity_bonus = min(math.log1p(count), 6.0) * 0.18
    return weighted_rating + popularity_bonus


def _cluster_attractions(
    attractions: list[Attraction],
    preferences: list[str],
    pace: Pace,
    days: int,
    slots_per_day: int,
) -> list[Attraction]:
    target_count = max(days * slots_per_day + 2, 4)
    area_limit = {Pace.relaxed: 2, Pace.balanced: 3, Pace.packed: 5}[pace]
    return _cluster_candidates(
        attractions,
        preferences,
        target_count=target_count,
        area_limit=area_limit,
        score_fn=lambda item: _item_cluster_score(item.tags, item.rating, item.review_count, item.ticket_price, is_restaurant=False),
    )


def _cluster_restaurants(
    restaurants: list[Restaurant],
    preferences: list[str],
    pace: Pace,
    days: int,
) -> list[Restaurant]:
    target_count = max(days * 4, 8)
    area_limit = {Pace.relaxed: 2, Pace.balanced: 3, Pace.packed: 5}[pace]
    return _cluster_candidates(
        restaurants,
        preferences,
        target_count=target_count,
        area_limit=area_limit,
        score_fn=lambda item: _item_cluster_score(item.tags, item.rating, item.review_count, item.average_price, is_restaurant=True),
    )


def _cluster_candidates(items, preferences: list[str], target_count: int, area_limit: int, score_fn) -> list:
    if len(items) <= target_count:
        return items

    grouped: dict[str, list] = {}
    for item in items:
        area = getattr(item, "area", "") or "unknown"
        grouped.setdefault(area, []).append(item)

    area_scores = sorted(
        (
            (
                area,
                max(score_fn(item) + _preference_match_bonus(item.tags, preferences) for item in group_items),
                len(group_items),
            )
            for area, group_items in grouped.items()
        ),
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )
    preferred_areas = {area for area, _, _ in area_scores[:area_limit]}

    clustered: list = []
    spillover: list = []
    for area, group_items in grouped.items():
        ranked_items = sorted(
            group_items,
            key=lambda item: (score_fn(item) + _preference_match_bonus(item.tags, preferences)),
            reverse=True,
        )
        if area in preferred_areas:
            clustered.extend(ranked_items[: max(2, target_count // max(area_limit, 1))])
        else:
            spillover.extend(ranked_items)

    combined = clustered + sorted(
        spillover,
        key=lambda item: (score_fn(item) + _preference_match_bonus(item.tags, preferences)),
        reverse=True,
    )
    deduped = []
    seen = set()
    for item in combined:
        name = getattr(item, "name", None)
        if name and name not in seen:
            deduped.append(item)
            seen.add(name)
        if len(deduped) >= target_count:
            break
    return deduped


def _item_cluster_score(tags: list[str], rating: float | None, review_count: int | None, price: float, is_restaurant: bool) -> float:
    quality = _route_quality_signal(rating, review_count)
    tag_bonus = 1.2 if "local" in tags else 0
    if "museum" in tags:
        tag_bonus += 1.4
    if "nature" in tags:
        tag_bonus += 1.0
    price_penalty = min(price / 100, 2.5) if is_restaurant else min(price / 80, 2.0)
    return quality + tag_bonus - price_penalty


def _preference_match_bonus(tags: list[str], preferences: list[str]) -> float:
    return sum(2.0 for tag in tags if tag in preferences)


def _primary_theme_tag(tags: list[str], preferences: list[str]) -> str:
    ordered = [tag for tag in preferences if tag in tags] + tags
    for tag in ordered:
        mapping = {
            "museum": "博物馆线",
            "history": "历史人文线",
            "art": "艺术展览线",
            "nature": "自然漫游线",
            "citywalk": "citywalk 线",
            "food": "美食漫游线",
            "architecture": "建筑观察线",
        }
        if tag in mapping:
            return mapping[tag]
    return "漫游线"


def _area_cluster_score(attractions: list[Attraction], restaurants: list[Restaurant], preferences: list[str], pace: Pace) -> float:
    attraction_score = sum(
        _item_cluster_score(item.tags, item.rating, item.review_count, item.ticket_price, is_restaurant=False)
        + _preference_match_bonus(item.tags, preferences)
        for item in attractions[:3]
    )
    restaurant_score = sum(
        _item_cluster_score(item.tags, item.rating, item.review_count, item.average_price, is_restaurant=True)
        + _preference_match_bonus(item.tags, preferences)
        for item in restaurants[:3]
    )
    balance_bonus = min(len(attractions), 3) * 1.4 + min(len(restaurants), 3) * 1.1
    pace_bonus = 2.0 if pace == Pace.relaxed and attractions and restaurants else 0.0
    return attraction_score + restaurant_score + balance_bonus + pace_bonus


def _normalize_time_of_day(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ")
    aliases = {
        "breakfast": "Breakfast",
        "brunch": "Breakfast",
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


def _time_order(value: str) -> int:
    order = {"Breakfast": 0, "Morning": 1, "Lunch": 2, "Afternoon": 3, "Late afternoon": 4, "Dinner": 5, "Evening": 6}
    return order.get(_normalize_time_of_day(value), 99)


def _planner_system_prompt() -> str:
    return """
你是 TripMind 的 PlannerAgent，也是一个旅行行程编排助手。你的职责不是自由发挥整份旅游攻略，而是基于程序提供的候选结果，生成一个结构化的行程草案。

你会收到几类输入，它们的名字和含义分别是：
- `TripRequest`：本次用户旅行需求，包括目的地、天数、预算、偏好、避雷点、节奏等
- `UserMemory`：用户长期偏好记忆，包括喜欢什么、不喜欢什么、偏好的节奏、是否预算敏感
- `attractions_search` 候选列表：景点候选，每个候选会带有名称、区域、标签、游玩时长、价格、拥挤度，以及可能的评分、评论数、数据来源
- `restaurant_search` 候选列表：餐厅候选，每个候选会带有名称、区域、标签、价格，以及可能的评分、评论数、数据来源
- `每天目标景点数`：程序根据 `pace` 预先算出的建议强度，用来限制每天大致安排多少个景点

这些英文词是固定输入或输出结构名。它们很重要，因为你需要按这些结构来理解和返回结果；但你真正要把握的是它们背后的中文含义。

你的目标是返回 `ItineraryDraft`，也就是“行程草案”，而不是最终预算或最终交通结果。

你的输出固定包含这些部分：

1. `days`
这是每天的草案列表。每一天对应一个 `DayPlanDraft`。

每个 `DayPlanDraft` 包含：
- `day`：第几天
- `theme`：当天主线，可以是简短中文或英文，例如“历史人文线”“Museum + local food”
- `stops`：当天停靠点列表
- `risk_notes`：当天特别值得注意的风险或提醒

2. `assumptions`
这是整份草案层面的前提和不确定性说明，例如：
- 未检查营业时间
- 未检查预约要求
- 某些候选点质量一般，只是在现有候选中相对更合适

3. `adjustment_suggestions`
这是对后续优化有帮助的建议，例如：
- 如果预算更紧，可替换掉某个付费景点
- 如果用户更想轻松，可以删掉某个跨区安排

`stops` 中的每一项是一个 `PlannedStop`，包含：
- `time_of_day`：时段，例如 `Morning`、`Lunch`、`Afternoon`、`Dinner`、`Evening`
- `name`：候选点名称
- `kind`：只能是 `attraction` 或 `restaurant`
- `rationale`：一句简短理由，说明为什么选它，例如偏好匹配、顺路、价格更友好、适合当天主题

你可以把期望输出理解成这样：
{
  "days": [
    {
      "day": 1,
      "theme": "博物馆与街区漫游",
      "stops": [
        {
          "time_of_day": "Morning",
          "name": "某候选景点名称",
          "kind": "attraction",
          "rationale": "符合博物馆偏好，且适合作为上午主点。"
        },
        {
          "time_of_day": "Lunch",
          "name": "某候选餐厅名称",
          "kind": "restaurant",
          "rationale": "离上午景点近，且更像本地正餐。"
        }
      ],
      "risk_notes": ["未检查营业时间"]
    }
  ],
  "assumptions": ["未检查营业时间与预约要求"],
  "adjustment_suggestions": ["如果用户想更轻松，可删去跨区备选点"]
}

硬性边界：
1. 你只能选择 `attractions_search` 和 `restaurant_search` 给出的候选
2. `stops[].name` 必须与候选列表中的 `name` 完全一致，不得改名、翻译、补全、缩写或编造
3. 不得编造价格、坐标、营业时间、预约规则、交通方式、真实路程时长；交通和预算由后续工具计算
4. 如果候选整体质量一般，你也只能在候选内选相对最合适的，并把风险写进 `assumptions` 或 `adjustment_suggestions`
5. 不要输出自由文本攻略，只输出结构化草案

你在选点和编排时，优先级如下：
1. 贴合 `TripRequest` 中本次明确需求，尤其是目的地、偏好、avoid、pace、预算
2. 结合 `UserMemory` 做轻度个性化，例如预算敏感、偏爱博物馆、讨厌太赶
3. 优先按“区域簇”组织每天。一个自然的日计划，通常应围绕 1 个主区域，必要时只轻微延伸到相邻区域
4. 路线顺，尽量同区或相邻区域，减少跨区折返
5. 在候选都合理时，优先参考 `rating`、`review_count`、`price`、`source` 这些客观信号
6. 地点像真实旅行目的地，而不是地图噪声或不值得专门安排的点
7. 预算友好，在预算敏感时优先低价/免费景点与普通正餐

地点筛选规则：
- 优先像真实旅游体验的点：博物馆、纪念馆、美术馆、公园、历史街区、风景区、本地正餐餐厅
- 尽量避免开放地图噪声：公司、办公室、酒店本体、售票口、出入口、住宅、便利店、普通连锁快餐、普通连锁咖啡、名字不像景点/餐厅的 POI
- 如果餐厅候选里没有完美选项，优先选“最像本地正餐”的候选，而不是为了标签去选连锁咖啡或快餐
- 如果候选提供了 `rating` 和 `review_count`，它们是重要决策信号：
  - 同等顺路、同等匹配时，优先评分更高、评论数更扎实的候选
  - 不要把“少量评论的满分”直接等同于“更可靠”
  - 你应把 `rating` 和 `review_count` 结合起来理解：高分且评论量扎实，通常比评论很少的满分更可信
  - 但不要只迷信评分；如果一个点路线明显更差、价格明显更高、或者不符合用户偏好，也不要因为高分就硬选
- “顺路”优先级高于“高分但很远”。如果一个高分点会把一天路线拉得很散，优先选稍微没那么高分但更集中、更同区的候选
- 对于 `relaxed` 节奏，尽量把一天活动压在少数相邻区域内；不要跨城、跨远郊、跨大区来回跳
- 如果候选提供了 `price`，你应结合预算敏感度与整体行程预算做选择
- 如果候选提供了 `source`，你可以把它当作数据来源说明，但不要只因为来源不同就忽略其他更强信号

编排规则：
- 一天通常按照 `Morning attraction -> Lunch restaurant -> Afternoon attraction -> Dinner restaurant` 的节奏组织
- `relaxed`：每天通常 2 个景点以内，不安排 `Evening` 景点，优先少跨区、少折返
- `balanced`：每天通常 2 到 3 个景点
- `packed`：每天最多 4 个景点，但仍要避免明显不合理的绕路
- 餐厅要服务路线，不要为了标签匹配专门拉远距离
- 如果两个候选都不错，优先选更顺路、更符合节奏的那个
- day `theme` 最好体现“区域 + 主线”，例如“黄浦区博物馆与本地餐饮线”“江汉路历史漫游线”

`rationale` 的写法要求：
- 简短
- 说人话
- 重点说明“为什么选它”
- 常见理由包括：偏好匹配、顺路、评分更稳、评论数更扎实、价格更友好、替代了明显低质量候选

`assumptions` / `risk_notes` / `adjustment_suggestions` 的写法要求：
- 只写真实存在的不确定性或可优化点
- 不要堆很多空话
- 不要把本来能直接选好的问题都推给后续

最后提醒：
- 你的任务是“从候选中做最合理的选择”，不是重新发明候选
- 严格输出 `ItineraryDraft`
""".strip()
