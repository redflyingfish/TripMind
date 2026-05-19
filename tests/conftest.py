import pytest

import tripmind.agents.planner as planner_module
from tripmind.schemas import Attraction, Restaurant, TransitEstimate
from tripmind.travel_data import estimate_budget


def fake_attractions_search(destination, preferences, avoid=None, max_results=8):
    return [
        Attraction(name="上海博物馆", area="人民广场", tags=["museum", "history", "art"], duration_minutes=120, ticket_price=0, crowd_level=3, latitude=31.2304, longitude=121.4700),
        Attraction(name="中华艺术宫", area="浦东", tags=["museum", "art"], duration_minutes=110, ticket_price=20, crowd_level=2, latitude=31.1869, longitude=121.4896),
        Attraction(name="上海自然博物馆", area="静安", tags=["museum", "science"], duration_minutes=130, ticket_price=30, crowd_level=4, latitude=31.2367, longitude=121.4626),
        Attraction(name="豫园", area="老城厢", tags=["history", "garden", "food"], duration_minutes=100, ticket_price=40, crowd_level=5, latitude=31.2272, longitude=121.4921),
    ][:max_results]


def fake_restaurant_search(destination, preferences, avoid=None, max_results=6):
    return [
        Restaurant(name="大壶春", area="人民广场", tags=["food", "local"], average_price=35, latitude=31.231, longitude=121.473),
        Restaurant(name="阿娘面馆", area="思南路", tags=["food", "noodle"], average_price=45, latitude=31.216, longitude=121.465),
        Restaurant(name="兰心餐厅", area="淮海路", tags=["food", "local"], average_price=120, latitude=31.221, longitude=121.461),
    ][:max_results]


def fake_estimate_transit(origin, destination, pace, **kwargs):
    return TransitEstimate(origin=origin, destination=destination, minutes=20, cost=8, mode="test", distance_km=2)


@pytest.fixture(autouse=True)
def patch_planner_tools(monkeypatch):
    monkeypatch.setattr(planner_module, "attractions_search", fake_attractions_search)
    monkeypatch.setattr(planner_module, "restaurant_search", fake_restaurant_search)
    monkeypatch.setattr(planner_module, "estimate_transit", fake_estimate_transit)
    monkeypatch.setattr(planner_module, "estimate_budget", estimate_budget)

