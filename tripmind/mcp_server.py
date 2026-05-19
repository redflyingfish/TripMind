from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tripmind import travel_data
from tripmind.schemas import Pace


mcp = FastMCP("tripmind-travel-tools", log_level="ERROR")


@mcp.tool()
def attractions_search(destination: str, preferences: list[str], avoid: list[str] | None = None, max_results: int = 8) -> list[dict]:
    """Search real OSM/Overpass attraction and cultural POI data."""
    return [item.model_dump(mode="json") for item in travel_data.attractions_search(destination, preferences, avoid, max_results)]


@mcp.tool()
def restaurant_search(destination: str, preferences: list[str], avoid: list[str] | None = None, max_results: int = 6) -> list[dict]:
    """Search real OSM/Overpass restaurant and cafe data."""
    return [item.model_dump(mode="json") for item in travel_data.restaurant_search(destination, preferences, avoid, max_results)]


@mcp.tool()
def estimate_transit(
    origin: str,
    destination: str,
    pace: str = "balanced",
    origin_latitude: float | None = None,
    origin_longitude: float | None = None,
    destination_latitude: float | None = None,
    destination_longitude: float | None = None,
) -> dict:
    """Estimate route pressure using coordinates and city-distance heuristics."""
    return travel_data.estimate_transit(
        origin=origin,
        destination=destination,
        pace=Pace(pace),
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude,
    ).model_dump(mode="json")


@mcp.tool()
def estimate_budget(
    attraction_costs: list[float],
    restaurant_costs: list[float],
    transit_costs: list[float],
    travelers: int,
    currency: str = "CNY",
    buffer_ratio: float = 0.12,
) -> dict:
    """Estimate itinerary budget from tool-selected components."""
    return travel_data.estimate_budget(
        attraction_costs=attraction_costs,
        restaurant_costs=restaurant_costs,
        transit_costs=transit_costs,
        travelers=travelers,
        currency=currency,
        buffer_ratio=buffer_ratio,
    ).model_dump(mode="json")


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
