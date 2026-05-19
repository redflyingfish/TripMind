from tripmind.agents.planner import PlannerAgent, _cluster_attractions
from tripmind.schemas import Attraction, ItineraryDraft, DayPlanDraft, Pace, PlannedStop, Restaurant, TripRequest, UserMemory


class DraftLLM:
    model = "draft-llm"

    def parse(self, *, system: str, user: str, schema):
        return ItineraryDraft(
            days=[
                DayPlanDraft(
                    day=1,
                    theme="Route test",
                    stops=[
                        PlannedStop(time_of_day="Morning", name="近处博物馆", kind="attraction"),
                        PlannedStop(time_of_day="Lunch", name="近处面馆", kind="restaurant"),
                        PlannedStop(time_of_day="Afternoon", name="远郊高分美术馆", kind="attraction"),
                        PlannedStop(time_of_day="Dinner", name="远郊酒楼", kind="restaurant"),
                    ],
                )
            ]
        )


def test_planner_compacts_overly_distant_llm_stops(monkeypatch) -> None:
    import tripmind.agents.planner as planner_module

    attractions = [
        Attraction(name="近处博物馆", area="中心区", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=3, latitude=31.2300, longitude=121.4700, source="baidu_place", rating=4.5, review_count=120),
        Attraction(name="近处美术馆", area="中心区", tags=["museum", "art"], duration_minutes=120, ticket_price=25, crowd_level=3, latitude=31.2320, longitude=121.4720, source="baidu_place", rating=4.3, review_count=60),
        Attraction(name="远郊高分美术馆", area="远郊", tags=["museum", "art"], duration_minutes=120, ticket_price=0, crowd_level=2, latitude=31.9000, longitude=121.9000, source="baidu_place", rating=5.0, review_count=1000),
    ]
    restaurants = [
        Restaurant(name="近处面馆", area="中心区", tags=["food", "local"], average_price=35, latitude=31.2310, longitude=121.4710, source="baidu_place", rating=4.4, review_count=80),
        Restaurant(name="近处本帮菜", area="中心区", tags=["food", "local"], average_price=68, latitude=31.2330, longitude=121.4730, source="baidu_place", rating=4.2, review_count=55),
        Restaurant(name="远郊酒楼", area="远郊", tags=["food", "local"], average_price=88, latitude=31.9100, longitude=121.9100, source="baidu_place", rating=5.0, review_count=600),
    ]

    monkeypatch.setattr(planner_module, "attractions_search", lambda **kwargs: attractions)
    monkeypatch.setattr(planner_module, "restaurant_search", lambda **kwargs: restaurants)

    request = TripRequest(destination="上海", days=1, preferences=["museum"], pace=Pace.relaxed)
    itinerary = PlannerAgent(llm=DraftLLM()).plan(request, UserMemory(user_id="u1"))

    titles = [item.title for item in itinerary.days[0].items]
    assert "近处美术馆" in titles
    assert "远郊高分美术馆" not in titles
    assert "近处本帮菜" in titles
    assert "远郊酒楼" not in titles


def test_cluster_attractions_prefers_stronger_areas() -> None:
    attractions = [
        Attraction(name="A区博物馆1", area="A区", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=3, source="baidu_place", rating=4.7, review_count=300),
        Attraction(name="A区博物馆2", area="A区", tags=["museum", "art"], duration_minutes=120, ticket_price=30, crowd_level=3, source="baidu_place", rating=4.6, review_count=220),
        Attraction(name="B区博物馆1", area="B区", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=3, source="baidu_place", rating=4.5, review_count=180),
        Attraction(name="B区公园", area="B区", tags=["nature"], duration_minutes=90, ticket_price=0, crowd_level=2, source="baidu_place", rating=4.4, review_count=140),
        Attraction(name="远郊冷门馆", area="远郊", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=2, source="baidu_place", rating=5.0, review_count=3),
        Attraction(name="远郊景点", area="远郊", tags=["sightseeing"], duration_minutes=90, ticket_price=10, crowd_level=2, source="baidu_place", rating=4.9, review_count=2),
    ]

    clustered = _cluster_attractions(attractions, preferences=["museum"], pace=Pace.relaxed, days=1, slots_per_day=2)
    names = [item.name for item in clustered]

    assert "A区博物馆1" in names
    assert "A区博物馆2" in names
    assert "B区博物馆1" in names or "B区公园" in names
    assert "远郊冷门馆" not in names


def test_planner_avoids_duplicate_titles_across_days(monkeypatch) -> None:
    import tripmind.agents.planner as planner_module

    attractions = [
        Attraction(name="博物馆A", area="中心区", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=3, latitude=31.2300, longitude=121.4700, source="baidu_place", rating=4.6, review_count=120),
        Attraction(name="博物馆B", area="中心区", tags=["museum", "art"], duration_minutes=110, ticket_price=25, crowd_level=3, latitude=31.2320, longitude=121.4720, source="baidu_place", rating=4.5, review_count=98),
        Attraction(name="公园C", area="静安", tags=["nature"], duration_minutes=90, ticket_price=0, crowd_level=2, latitude=31.2350, longitude=121.4680, source="baidu_place", rating=4.4, review_count=86),
        Attraction(name="历史馆D", area="黄浦", tags=["museum", "history"], duration_minutes=100, ticket_price=15, crowd_level=3, latitude=31.2250, longitude=121.4800, source="baidu_place", rating=4.3, review_count=76),
    ]
    restaurants = [
        Restaurant(name="面馆A", area="中心区", tags=["food", "local"], average_price=35, latitude=31.2310, longitude=121.4710, source="baidu_place", rating=4.4, review_count=80),
        Restaurant(name="本帮菜B", area="中心区", tags=["food", "local"], average_price=68, latitude=31.2330, longitude=121.4730, source="baidu_place", rating=4.2, review_count=55),
        Restaurant(name="小吃C", area="静安", tags=["food", "local"], average_price=42, latitude=31.2360, longitude=121.4670, source="baidu_place", rating=4.3, review_count=64),
        Restaurant(name="酒家D", area="黄浦", tags=["food", "local"], average_price=72, latitude=31.2260, longitude=121.4810, source="baidu_place", rating=4.5, review_count=70),
    ]

    monkeypatch.setattr(planner_module, "attractions_search", lambda **kwargs: attractions)
    monkeypatch.setattr(planner_module, "restaurant_search", lambda **kwargs: restaurants)

    request = TripRequest(destination="上海", days=2, preferences=["museum", "food"], pace=Pace.relaxed)
    itinerary = PlannerAgent().plan(request, UserMemory(user_id="u1"))

    titles = [item.title for day in itinerary.days for item in day.items]
    assert len(titles) == len(set(titles))


def test_planner_respects_requested_time_slots(monkeypatch) -> None:
    import tripmind.agents.planner as planner_module

    attractions = [
        Attraction(name="博物馆A", area="中心区", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=3, latitude=31.2300, longitude=121.4700, source="baidu_place", rating=4.6, review_count=120),
        Attraction(name="公园B", area="静安", tags=["nature"], duration_minutes=90, ticket_price=0, crowd_level=2, latitude=31.2350, longitude=121.4680, source="baidu_place", rating=4.4, review_count=86),
    ]
    restaurants = [
        Restaurant(name="晚餐A", area="中心区", tags=["food", "local"], average_price=68, latitude=31.2330, longitude=121.4730, source="baidu_place", rating=4.2, review_count=55),
    ]

    monkeypatch.setattr(planner_module, "attractions_search", lambda **kwargs: attractions)
    monkeypatch.setattr(planner_module, "restaurant_search", lambda **kwargs: restaurants)

    request = TripRequest(
        destination="上海",
        days=1,
        preferences=["museum"],
        pace=Pace.relaxed,
        requested_time_slots=["Afternoon", "Dinner", "Evening"],
        avoid_time_slots=["Morning"],
    )
    itinerary = PlannerAgent().plan(request, UserMemory(user_id="u1"))

    slots = [item.time_of_day for item in itinerary.days[0].items]
    assert "Morning" not in slots
    assert "Afternoon" in slots
    assert "Dinner" in slots
    assert "Evening" in slots


def test_planner_supports_breakfast_when_explicitly_requested(monkeypatch) -> None:
    import tripmind.agents.planner as planner_module

    attractions = [
        Attraction(name="博物馆A", area="中心区", tags=["museum"], duration_minutes=120, ticket_price=20, crowd_level=3, latitude=31.2300, longitude=121.4700, source="baidu_place", rating=4.6, review_count=120),
        Attraction(name="夜景B", area="外滩", tags=["citywalk"], duration_minutes=90, ticket_price=0, crowd_level=2, latitude=31.2400, longitude=121.4900, source="baidu_place", rating=4.5, review_count=140),
    ]
    restaurants = [
        Restaurant(name="早餐A", area="中心区", tags=["food", "local"], average_price=22, latitude=31.2310, longitude=121.4710, source="baidu_place", rating=4.4, review_count=80),
        Restaurant(name="晚餐A", area="外滩", tags=["food", "local"], average_price=68, latitude=31.2410, longitude=121.4910, source="baidu_place", rating=4.2, review_count=55),
    ]

    monkeypatch.setattr(planner_module, "attractions_search", lambda **kwargs: attractions)
    monkeypatch.setattr(planner_module, "restaurant_search", lambda **kwargs: restaurants)

    request = TripRequest(
        destination="上海",
        days=1,
        preferences=["food"],
        pace=Pace.relaxed,
        requested_time_slots=["Breakfast", "Afternoon", "Dinner", "Evening"],
    )
    itinerary = PlannerAgent().plan(request, UserMemory(user_id="u1"))

    slots = [item.time_of_day for item in itinerary.days[0].items]
    assert "Breakfast" in slots
    assert any(item.kind == "restaurant" and item.time_of_day == "Breakfast" for item in itinerary.days[0].items)


def test_plan_variants_are_sorted_by_actual_total(monkeypatch) -> None:
    import tripmind.agents.planner as planner_module

    attractions = [
        Attraction(name="免费公园", area="中心区", tags=["nature"], duration_minutes=90, ticket_price=0, crowd_level=2, latitude=31.23, longitude=121.47, source="baidu_place", rating=4.2, review_count=80),
        Attraction(name="收费博物馆", area="中心区", tags=["museum"], duration_minutes=120, ticket_price=80, crowd_level=3, latitude=31.24, longitude=121.48, source="baidu_place", rating=4.8, review_count=200),
        Attraction(name="艺术馆", area="中心区", tags=["art"], duration_minutes=100, ticket_price=50, crowd_level=3, latitude=31.25, longitude=121.49, source="baidu_place", rating=4.6, review_count=140),
    ]
    restaurants = [
        Restaurant(name="小馆子", area="中心区", tags=["food", "local"], average_price=25, latitude=31.231, longitude=121.471, source="baidu_place", rating=4.2, review_count=60),
        Restaurant(name="正餐馆", area="中心区", tags=["food", "local"], average_price=60, latitude=31.232, longitude=121.472, source="baidu_place", rating=4.4, review_count=90),
        Restaurant(name="体验餐厅", area="中心区", tags=["food", "local"], average_price=120, latitude=31.233, longitude=121.473, source="baidu_place", rating=4.7, review_count=150),
    ]

    monkeypatch.setattr(planner_module, "attractions_search", lambda **kwargs: attractions)
    monkeypatch.setattr(planner_module, "restaurant_search", lambda **kwargs: restaurants)

    request = TripRequest(destination="上海", days=1, preferences=["food"], pace=Pace.balanced)
    variants = PlannerAgent().plan_variants(request, UserMemory(user_id="u1"))

    totals = [itinerary.budget.total for _, _, itinerary in variants]
    labels = [label for _, label, _ in variants]
    assert totals == sorted(totals)
    assert labels[0] == "省钱版"
