from __future__ import annotations

from tripmind.mcp_client import McpTravelClient
from tripmind.schemas import Attraction, BudgetBreakdown, Pace, Restaurant, TransitEstimate


_client = McpTravelClient()


def attractions_search(
    destination: str,
    preferences: list[str],
    avoid: list[str] | None = None,
    max_results: int = 8,
) -> list[Attraction]:
    """Call TripMind's MCP attraction search tool."""
    data = _client.call_tool(
        "attractions_search",
        {"destination": destination, "preferences": preferences, "avoid": avoid or [], "max_results": max_results},
    )
    return [Attraction.model_validate(item) for item in data]


def restaurant_search(
    destination: str,
    preferences: list[str],
    avoid: list[str] | None = None,
    max_results: int = 6,
) -> list[Restaurant]:
    """Call TripMind's MCP restaurant search tool."""
    data = _client.call_tool(
        "restaurant_search",
        {"destination": destination, "preferences": preferences, "avoid": avoid or [], "max_results": max_results},
    )
    return [Restaurant.model_validate(item) for item in data]


def estimate_transit(
    origin: str,
    destination: str,
    pace: Pace = Pace.balanced,
    origin_latitude: float | None = None,
    origin_longitude: float | None = None,
    destination_latitude: float | None = None,
    destination_longitude: float | None = None,
) -> TransitEstimate:
    """Call TripMind's MCP transit pressure tool."""
    data = _client.call_tool(
        "estimate_transit",
        {
            "origin": origin,
            "destination": destination,
            "pace": pace.value,
            "origin_latitude": origin_latitude,
            "origin_longitude": origin_longitude,
            "destination_latitude": destination_latitude,
            "destination_longitude": destination_longitude,
        },
    )
    return TransitEstimate.model_validate(data)


def estimate_budget(
    attraction_costs: list[float],
    restaurant_costs: list[float],
    transit_costs: list[float],
    travelers: int,
    currency: str = "CNY",
    buffer_ratio: float = 0.12,
) -> BudgetBreakdown:
    """Call TripMind's MCP budget tool."""
    data = _client.call_tool(
        "estimate_budget",
        {
            "attraction_costs": attraction_costs,
            "restaurant_costs": restaurant_costs,
            "transit_costs": transit_costs,
            "travelers": travelers,
            "currency": currency,
            "buffer_ratio": buffer_ratio,
        },
    )
    return BudgetBreakdown.model_validate(data)

