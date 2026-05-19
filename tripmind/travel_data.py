from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any

import httpx
from dotenv import load_dotenv

from tripmind.schemas import Attraction, BudgetBreakdown, Pace, Restaurant, TransitEstimate


load_dotenv()


NOMINATIM_URL = os.getenv("TRIPMIND_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
OVERPASS_URLS = [
    url.strip()
    for url in os.getenv(
        "TRIPMIND_OVERPASS_URLS",
        "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter,https://overpass.openstreetmap.ru/api/interpreter",
    ).split(",")
    if url.strip()
]
USER_AGENT = os.getenv("TRIPMIND_USER_AGENT", "TripMind/0.1 open-source travel planner")
TRAVEL_PROVIDER = os.getenv("TRIPMIND_TRAVEL_PROVIDER", "amap").lower()
AMAP_API_KEY = os.getenv("AMAP_API_KEY") or os.getenv("GAODE_API_KEY")
AMAP_PLACE_URL = os.getenv("TRIPMIND_AMAP_PLACE_URL", "https://restapi.amap.com/v3/place/text")
AMAP_WALKING_URL = os.getenv("TRIPMIND_AMAP_WALKING_URL", "https://restapi.amap.com/v3/direction/walking")
AMAP_DRIVING_URL = os.getenv("TRIPMIND_AMAP_DRIVING_URL", "https://restapi.amap.com/v3/direction/driving")
BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK") or os.getenv("BAIDU_AK") or os.getenv("BAIDU_MAPS_AK")
BAIDU_PLACE_REGION_URL = os.getenv("TRIPMIND_BAIDU_PLACE_REGION_URL", "https://api.map.baidu.com/place/v3/region")
BAIDU_PLACE_DETAIL_URL = os.getenv("TRIPMIND_BAIDU_PLACE_DETAIL_URL", "https://api.map.baidu.com/place/v3/detail")
BAIDU_GEOCODING_URL = os.getenv("TRIPMIND_BAIDU_GEOCODING_URL", "https://api.map.baidu.com/geocoding/v3/")
RATING_PRIOR_MEAN = 4.2
RATING_PRIOR_WEIGHT = 80
BAD_POI_KEYWORDS = [
    "公司",
    "有限公司",
    "办公室",
    "写字楼",
    "停车",
    "入口",
    "出口",
    "售票处",
    "服务中心",
    "管理处",
    "售楼处",
    "门店",
]
CHAIN_RESTAURANT_KEYWORDS = [
    "kfc",
    "肯德基",
    "mcdonald",
    "麦当劳",
    "starbucks",
    "星巴克",
    "吉野家",
    "85度",
    "85°",
    "必胜客",
    "pizza hut",
    "burger king",
    "汉堡王",
]
BAD_RESTAURANT_KEYWORDS = [
    "酒店",
    "宾馆",
    "大厦",
    "便利店",
    "超市",
    "食堂",
    "canteen",
    "国际饭店",
]
BEVERAGE_VENUE_KEYWORDS = [
    "咖啡",
    "coffee",
    "cafe",
    "酒吧",
    "bar",
    "taproom",
    "brew",
    "club",
]
LAST_PROVIDER_WARNINGS: list[str] = []


def consume_provider_warnings() -> list[str]:
    global LAST_PROVIDER_WARNINGS
    warnings = LAST_PROVIDER_WARNINGS[:]
    LAST_PROVIDER_WARNINGS = []
    return warnings


def _remember_provider_warning(message: str) -> None:
    global LAST_PROVIDER_WARNINGS
    if message not in LAST_PROVIDER_WARNINGS:
        LAST_PROVIDER_WARNINGS.append(message)


@lru_cache(maxsize=128)
def geocode_place(place: str) -> tuple[float, float]:
    if _use_baidu():
        try:
            return _baidu_geocode_place(place)
        except RuntimeError:
            pass
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(NOMINATIM_URL, params={"q": place, "format": "json", "limit": 1})
        response.raise_for_status()
        data = response.json()
    if not data:
        raise RuntimeError(f"Could not geocode destination: {place}")
    return float(data[0]["lat"]), float(data[0]["lon"])


def attractions_search(
    destination: str,
    preferences: list[str],
    avoid: list[str] | None = None,
    max_results: int = 8,
) -> list[Attraction]:
    if _use_baidu():
        try:
            return _baidu_attractions_search(destination, preferences, avoid or [], max_results)
        except RuntimeError as exc:
            _remember_provider_warning(f"Baidu attraction search failed; fell back to open data. Reason: {exc}")
            pass
    if _use_amap():
        try:
            return _amap_attractions_search(destination, preferences, avoid or [], max_results)
        except RuntimeError as exc:
            _remember_provider_warning(f"Amap attraction search failed; fell back to open data. Reason: {exc}")
            pass

    lat, lon = geocode_place(destination)
    try:
        elements = _overpass_search(
            lat=lat,
            lon=lon,
            radius_m=9000,
            filters=[
                'node["tourism"~"museum|gallery|attraction|artwork"]',
                'way["tourism"~"museum|gallery|attraction|artwork"]',
                'relation["tourism"~"museum|gallery|attraction|artwork"]',
                'node["historic"]',
                'way["historic"]',
                'node["leisure"~"park|garden"]',
                'way["leisure"~"park|garden"]',
            ],
        )
    except RuntimeError:
        return _nominatim_attractions(destination, preferences, avoid or [], max_results, lat, lon)
    items: list[Attraction] = []
    seen: set[str] = set()
    for element in elements:
        tags = element.get("tags") or {}
        name = _name(tags)
        if not name or name in seen:
            continue
        if _bad_attraction_name(name) or not _looks_like_attraction_name(name):
            continue
        point = _point(element)
        if point is None:
            continue
        item_tags = _attraction_tags(tags)
        if _is_avoided(item_tags, avoid or []):
            continue
        items.append(
                Attraction(
                    name=name,
                    area=_area(tags, destination),
                    tags=item_tags,
                    duration_minutes=_duration_for_attraction(item_tags),
                    ticket_price=_ticket_price(item_tags),
                    crowd_level=_crowd_level(tags, item_tags),
                    source="osm_overpass",
                    latitude=point[0],
                    longitude=point[1],
                )
        )
        seen.add(name)
    return _rank_attractions(items, preferences, avoid or [], lat, lon)[:max_results]


def restaurant_search(
    destination: str,
    preferences: list[str],
    avoid: list[str] | None = None,
    max_results: int = 6,
) -> list[Restaurant]:
    if _use_baidu():
        try:
            return _baidu_restaurant_search(destination, preferences, avoid or [], max_results)
        except RuntimeError as exc:
            _remember_provider_warning(f"Baidu restaurant search failed; fell back to open data. Reason: {exc}")
            pass
    if _use_amap():
        try:
            return _amap_restaurant_search(destination, preferences, avoid or [], max_results)
        except RuntimeError as exc:
            _remember_provider_warning(f"Amap restaurant search failed; fell back to open data. Reason: {exc}")
            pass

    lat, lon = geocode_place(destination)
    try:
        elements = _overpass_search(
            lat=lat,
            lon=lon,
            radius_m=9000,
            filters=[
                'node["amenity"="restaurant"]',
                'way["amenity"="restaurant"]',
            ],
        )
    except RuntimeError:
        return _nominatim_restaurants(destination, preferences, avoid or [], max_results, lat, lon)
    items: list[Restaurant] = []
    seen: set[str] = set()
    for element in elements:
        tags = element.get("tags") or {}
        name = _name(tags)
        if not name or name in seen:
            continue
        if _bad_restaurant_name(name):
            continue
        if _is_beverage_first_venue(name, preferences):
            continue
        point = _point(element)
        if point is None:
            continue
        item_tags = _restaurant_tags(tags)
        if _is_avoided(item_tags, avoid or []):
            continue
        items.append(
                Restaurant(
                    name=name,
                    area=_area(tags, destination),
                    tags=item_tags,
                    average_price=_restaurant_price(item_tags),
                    source="osm_overpass",
                    latitude=point[0],
                    longitude=point[1],
                )
        )
        seen.add(name)
    return _rank_restaurants(items, preferences, avoid or [], lat, lon)[:max_results]


def estimate_transit(
    origin: str,
    destination: str,
    pace: Pace = Pace.balanced,
    origin_latitude: float | None = None,
    origin_longitude: float | None = None,
    destination_latitude: float | None = None,
    destination_longitude: float | None = None,
) -> TransitEstimate:
    if origin_latitude is None or origin_longitude is None or destination_latitude is None or destination_longitude is None:
        origin_latitude, origin_longitude = geocode_place(origin)
        destination_latitude, destination_longitude = geocode_place(destination)

    if _use_amap():
        try:
            return _amap_transit_estimate(
                origin,
                destination,
                pace,
                origin_latitude,
                origin_longitude,
                destination_latitude,
                destination_longitude,
            )
        except RuntimeError:
            pass

    distance_km = _haversine_km(origin_latitude, origin_longitude, destination_latitude, destination_longitude)
    if distance_km < 0.35:
        return TransitEstimate(origin=origin, destination=destination, minutes=10, cost=0, mode="walk", distance_km=round(distance_km, 2))

    minutes = math.ceil(distance_km / 18 * 60 + 12)
    if pace == Pace.relaxed:
        minutes += 6
    elif pace == Pace.packed:
        minutes = max(10, minutes - 4)
    cost = max(4, round(distance_km * 2.6, 1))
    return TransitEstimate(origin=origin, destination=destination, minutes=minutes, cost=cost, mode="metro/taxi estimate", distance_km=round(distance_km, 2))


def estimate_budget(
    attraction_costs: list[float],
    restaurant_costs: list[float],
    transit_costs: list[float],
    travelers: int,
    currency: str = "CNY",
    buffer_ratio: float = 0.12,
) -> BudgetBreakdown:
    attractions = sum(attraction_costs) * travelers
    restaurants = sum(restaurant_costs) * travelers
    transit = sum(transit_costs) * travelers
    subtotal = attractions + restaurants + transit
    buffer = round(subtotal * buffer_ratio, 2)
    total = round(subtotal + buffer, 2)
    return BudgetBreakdown(
        attractions=round(attractions, 2),
        restaurants=round(restaurants, 2),
        transit=round(transit, 2),
        buffer=buffer,
        total=total,
        currency=currency,
    )


def _use_amap() -> bool:
    return TRAVEL_PROVIDER in {"amap", "gaode"} and bool(AMAP_API_KEY)


def _use_baidu() -> bool:
    return TRAVEL_PROVIDER == "baidu" and bool(BAIDU_MAP_AK)


def _baidu_geocode_place(place: str) -> tuple[float, float]:
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(
            BAIDU_GEOCODING_URL,
            params={"address": place, "output": "json", "ak": BAIDU_MAP_AK},
        )
        response.raise_for_status()
        data = response.json()
    if data.get("status") != 0:
        raise RuntimeError(f"Baidu geocoding failed: {data.get('msg') or data}")
    location = (data.get("result") or {}).get("location") or {}
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        raise RuntimeError(f"Baidu geocoding returned no location for: {place}")
    return float(lat), float(lng)


def _baidu_attractions_search(
    destination: str,
    preferences: list[str],
    avoid: list[str],
    max_results: int,
) -> list[Attraction]:
    queries = _baidu_attraction_queries(destination, preferences)
    center_lat, center_lon = geocode_place(destination)
    scored_items: list[tuple[Attraction, float, int]] = []
    seen: set[str] = set()
    for query in queries:
        results = _baidu_enrich_results(_baidu_region_search(query=query, region=destination, page_size=20))
        for poi in results:
            name = (poi.get("name") or "").strip()
            uid = poi.get("uid") or name
            if not name or uid in seen or _bad_attraction_name(name) or not _looks_like_attraction_name(name):
                continue
            point = _baidu_location(poi)
            if point is None:
                continue
            tags = _baidu_attraction_tags(poi)
            if _is_avoided(tags, avoid):
                continue
            scored_items.append(
                (
                    Attraction(
                        name=name,
                        area=_baidu_area(poi, destination),
                        tags=tags,
                        duration_minutes=_duration_for_attraction(tags),
                        ticket_price=_baidu_ticket_price(poi, tags),
                        crowd_level=_baidu_crowd_level(poi, tags),
                        source="baidu_place",
                        rating=_baidu_rating(poi) or None,
                        review_count=_baidu_comment_count(poi) or None,
                        latitude=point[0],
                        longitude=point[1],
                    ),
                    _baidu_rating(poi),
                    _baidu_comment_count(poi),
                )
            )
            seen.add(uid)
    ranked = sorted(
        scored_items,
        key=lambda item: (
            _score(item[0].tags, preferences, avoid),
            _quality_signal(item[1], item[2]),
            _attraction_name_score(item[0].name),
            -item[0].crowd_level,
            -_distance_from_center(item[0].latitude, item[0].longitude, center_lat, center_lon),
        ),
        reverse=True,
    )
    return [item for item, _, _ in ranked[:max_results]]


def _baidu_restaurant_search(
    destination: str,
    preferences: list[str],
    avoid: list[str],
    max_results: int,
) -> list[Restaurant]:
    queries = [f"{destination} 本地餐厅", f"{destination} 特色餐厅", f"{destination} 小吃", f"{destination} 美食"]
    center_lat, center_lon = geocode_place(destination)
    scored_items: list[tuple[Restaurant, float, int]] = []
    seen: set[str] = set()
    for query in queries:
        results = _baidu_enrich_results(
            _baidu_region_search(
                query=query,
                region=destination,
                page_size=20,
                filter_="industry_type:cater|sort_name:overall_rating|sort_rule:0",
            )
        )
        for poi in results:
            name = (poi.get("name") or "").strip()
            uid = poi.get("uid") or name
            if not name or uid in seen or _bad_restaurant_name(name):
                continue
            if _is_beverage_first_venue(name, preferences):
                continue
            point = _baidu_location(poi)
            if point is None:
                continue
            tags = _baidu_restaurant_tags(poi)
            if _is_avoided(tags, avoid):
                continue
            scored_items.append(
                (
                    Restaurant(
                        name=name,
                        area=_baidu_area(poi, destination),
                        tags=tags,
                        average_price=_baidu_restaurant_price(poi, tags),
                        source="baidu_place",
                        rating=_baidu_rating(poi) or None,
                        review_count=_baidu_comment_count(poi) or None,
                        latitude=point[0],
                        longitude=point[1],
                    ),
                    _baidu_rating(poi),
                    _baidu_comment_count(poi),
                )
            )
            seen.add(uid)
    ranked = sorted(
        scored_items,
        key=lambda item: (
            _score(item[0].tags, preferences, avoid),
            _quality_signal(item[1], item[2]),
            -10 if "chain" in item[0].tags else 0,
            4 if "local" in item[0].tags else 0,
            _restaurant_name_score(item[0].name),
            -_distance_from_center(item[0].latitude, item[0].longitude, center_lat, center_lon),
            -item[0].average_price,
        ),
        reverse=True,
    )
    return [item for item, _, _ in ranked[:max_results]]


def _baidu_region_search(
    query: str,
    region: str,
    page_size: int = 20,
    filter_: str | None = None,
) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "region": region,
        "output": "json",
        "ak": BAIDU_MAP_AK,
        "scope": 2,
        "page_size": max(10, min(page_size, 20)),
        "page_num": 0,
        "ret_coordtype": "gcj02ll",
        "extensions_adcode": "true",
    }
    if filter_:
        params["filter"] = filter_
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(BAIDU_PLACE_REGION_URL, params=params)
        response.raise_for_status()
        data = response.json()
    if data.get("status") != 0:
        raise RuntimeError(f"Baidu place search failed: {data.get('message') or data}")
    return data.get("results", [])


def _baidu_enrich_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing_uids = [item.get("uid") for item in results if item.get("uid") and not item.get("detail_info")]
    if not missing_uids:
        return results
    detail_map = _baidu_detail_lookup(missing_uids[:10])
    enriched: list[dict[str, Any]] = []
    for item in results:
        uid = item.get("uid")
        detail = detail_map.get(uid)
        if detail:
            merged = dict(item)
            detail_info = dict(item.get("detail_info") or {})
            detail_info.update(detail.get("detail_info") or {})
            merged["detail_info"] = detail_info
            if detail.get("location") and not merged.get("location"):
                merged["location"] = detail["location"]
            if detail.get("area") and not merged.get("area"):
                merged["area"] = detail["area"]
            if detail.get("address") and not merged.get("address"):
                merged["address"] = detail["address"]
            enriched.append(merged)
        else:
            enriched.append(item)
    return enriched


def _baidu_detail_lookup(uids: list[str]) -> dict[str, dict[str, Any]]:
    if not uids:
        return {}
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(
            BAIDU_PLACE_DETAIL_URL,
            params={
                "uids": ",".join(uids[:10]),
                "scope": 2,
                "output": "json",
                "ak": BAIDU_MAP_AK,
                "ret_coordtype": "gcj02ll",
                "extensions_adcode": "true",
            },
        )
        response.raise_for_status()
        data = response.json()
    if data.get("status") != 0:
        return {}
    result = data.get("result") or []
    if isinstance(result, dict):
        result = [result]
    return {item.get("uid"): item for item in result if item.get("uid")}


def _amap_attractions_search(
    destination: str,
    preferences: list[str],
    avoid: list[str],
    max_results: int,
) -> list[Attraction]:
    keywords = _amap_attraction_keywords(preferences)
    items: list[Attraction] = []
    seen: set[str] = set()
    for keyword in keywords:
        for poi in _amap_place_search(keyword=keyword, city=destination, types="110000|140000", offset=20):
            name = poi.get("name", "")
            if not name or name in seen or _bad_attraction_name(name) or not _looks_like_attraction_name(name):
                continue
            point = _amap_location(poi)
            if point is None:
                continue
            tags = _amap_attraction_tags(poi)
            if _is_avoided(tags, avoid):
                continue
            items.append(
                Attraction(
                    name=name,
                    area=poi.get("adname") or poi.get("pname") or destination,
                    tags=tags,
                    duration_minutes=_duration_for_attraction(tags),
                    ticket_price=_amap_ticket_price(poi, tags),
                    crowd_level=_amap_crowd_level(poi, tags),
                    source="amap_place",
                    rating=_amap_rating(poi) or None,
                    latitude=point[0],
                    longitude=point[1],
                )
            )
            seen.add(name)
    center_lat, center_lon = geocode_place(destination)
    return _rank_attractions(items, preferences, avoid, center_lat, center_lon)[:max_results]


def _amap_restaurant_search(
    destination: str,
    preferences: list[str],
    avoid: list[str],
    max_results: int,
) -> list[Restaurant]:
    keywords = [f"{destination} 本地餐厅", f"{destination} 特色餐厅", f"{destination} 餐厅", f"{destination} 小吃"]
    items: list[Restaurant] = []
    seen: set[str] = set()
    for keyword in keywords:
        for poi in _amap_place_search(keyword=keyword, city=destination, types="050000", offset=20):
            name = poi.get("name", "")
            if not name or name in seen or _bad_restaurant_name(name):
                continue
            if _is_beverage_first_venue(name, preferences):
                continue
            point = _amap_location(poi)
            if point is None:
                continue
            tags = _amap_restaurant_tags(poi)
            if _is_avoided(tags, avoid):
                continue
            items.append(
                Restaurant(
                    name=name,
                    area=poi.get("adname") or poi.get("pname") or destination,
                    tags=tags,
                    average_price=_amap_restaurant_price(poi, tags),
                    source="amap_place",
                    rating=_amap_rating(poi) or None,
                    latitude=point[0],
                    longitude=point[1],
                )
            )
            seen.add(name)
    center_lat, center_lon = geocode_place(destination)
    return _rank_restaurants(items, preferences, avoid, center_lat, center_lon)[:max_results]


def _amap_place_search(keyword: str, city: str, types: str, offset: int) -> list[dict[str, Any]]:
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(
            AMAP_PLACE_URL,
            params={
                "key": AMAP_API_KEY,
                "keywords": keyword,
                "city": city,
                "types": types,
                "citylimit": "true",
                "offset": offset,
                "page": 1,
                "extensions": "all",
            },
        )
        response.raise_for_status()
        data = response.json()
    if data.get("status") != "1":
        raise RuntimeError(f"Amap place search failed: {data.get('info') or data}")
    return data.get("pois", [])


def _amap_transit_estimate(
    origin: str,
    destination: str,
    pace: Pace,
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
) -> TransitEstimate:
    distance_km = _haversine_km(origin_latitude, origin_longitude, destination_latitude, destination_longitude)
    endpoint = AMAP_WALKING_URL if distance_km <= 1.5 else AMAP_DRIVING_URL
    mode = "amap walking" if endpoint == AMAP_WALKING_URL else "amap driving estimate"
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(
            endpoint,
            params={
                "key": AMAP_API_KEY,
                "origin": f"{origin_longitude},{origin_latitude}",
                "destination": f"{destination_longitude},{destination_latitude}",
                "extensions": "base",
            },
        )
        response.raise_for_status()
        data = response.json()
    if data.get("status") != "1":
        raise RuntimeError(f"Amap route failed: {data.get('info') or data}")
    path = (data.get("route", {}).get("paths") or [{}])[0]
    seconds = float(path.get("duration") or 0)
    meters = float(path.get("distance") or distance_km * 1000)
    minutes = max(8, math.ceil(seconds / 60)) if seconds else math.ceil(distance_km / 18 * 60 + 12)
    if pace == Pace.relaxed:
        minutes += 5
    elif pace == Pace.packed:
        minutes = max(8, minutes - 3)
    distance_km = meters / 1000
    cost = 0 if mode == "amap walking" else max(8, round(distance_km * 2.8, 1))
    return TransitEstimate(origin=origin, destination=destination, minutes=minutes, cost=cost, mode=mode, distance_km=round(distance_km, 2))


def _amap_location(poi: dict[str, Any]) -> tuple[float, float] | None:
    location = poi.get("location")
    if not location or "," not in location:
        return None
    lon, lat = location.split(",", 1)
    return float(lat), float(lon)


def _amap_attraction_keywords(preferences: list[str]) -> list[str]:
    terms = ["博物馆", "美术馆", "纪念馆", "公园", "景点"]
    if "nature" in preferences:
        terms = ["公园", "花园", *terms]
    if "history" in preferences:
        terms = ["纪念馆", "博物馆", *terms]
    if "art" in preferences:
        terms = ["美术馆", "艺术馆", *terms]
    return _unique(terms)


def _amap_attraction_tags(poi: dict[str, Any]) -> list[str]:
    text = f"{poi.get('name', '')} {poi.get('type', '')}"
    result: list[str] = []
    if _has_any(text, ["博物馆", "纪念馆", "展览馆", "陈列馆"]):
        result.extend(["museum", "history"])
    if _has_any(text, ["美术馆", "艺术馆", "画廊"]):
        result.extend(["museum", "art"])
    if _has_any(text, ["公园", "花园", "风景", "自然"]):
        result.extend(["nature", "relaxed"])
    if _has_any(text, ["风景名胜", "景点", "旅游景点"]):
        result.append("sightseeing")
    return _unique(result or ["sightseeing"])


def _amap_restaurant_tags(poi: dict[str, Any]) -> list[str]:
    text = f"{poi.get('name', '')} {poi.get('type', '')}".lower()
    result = ["food", "local"]
    cuisine_map = {
        "本帮": "shanghai",
        "江浙": "jiangzhe",
        "小吃": "snack",
        "面": "noodle",
        "火锅": "hotpot",
        "粤菜": "cantonese",
        "川菜": "sichuan",
    }
    for word, tag in cuisine_map.items():
        if word in text:
            result.append(tag)
    if _has_any(text, CHAIN_RESTAURANT_KEYWORDS):
        result.append("chain")
    return _unique(result)


def _amap_ticket_price(poi: dict[str, Any], tags: list[str]) -> float:
    cost = ((poi.get("biz_ext") or {}).get("cost") or "").strip()
    if cost:
        try:
            return float(cost)
        except ValueError:
            pass
    return _ticket_price(tags)


def _amap_restaurant_price(poi: dict[str, Any], tags: list[str]) -> float:
    cost = ((poi.get("biz_ext") or {}).get("cost") or "").strip()
    if cost:
        try:
            return float(cost)
        except ValueError:
            pass
    return _restaurant_price(tags)


def _amap_crowd_level(poi: dict[str, Any], tags: list[str]) -> int:
    rating = _amap_rating(poi)
    if rating:
        return 4 if rating >= 4.5 else 3
    return _crowd_level({}, tags)


def _amap_rating(poi: dict[str, Any]) -> float:
    rating = ((poi.get("biz_ext") or {}).get("rating") or "").strip()
    if rating:
        try:
            return float(rating)
        except ValueError:
            return 0.0
    return 0.0


def _overpass_search(lat: float, lon: float, radius_m: int, filters: list[str]) -> list[dict[str, Any]]:
    query_parts = "\n".join(f"{filter_}(around:{radius_m},{lat},{lon});" for filter_ in filters)
    query = f"""
    [out:json][timeout:25];
    (
      {query_parts}
    );
    out center tags 80;
    """
    last_error: Exception | None = None
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        for url in OVERPASS_URLS:
            try:
                response = client.post(url, data={"data": query})
                response.raise_for_status()
                return response.json().get("elements", [])
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
    raise RuntimeError(f"Overpass search failed after {len(OVERPASS_URLS)} endpoint(s): {last_error}")


def _nominatim_attractions(
    destination: str,
    preferences: list[str],
    avoid: list[str],
    max_results: int,
    center_lat: float,
    center_lon: float,
) -> list[Attraction]:
    queries = [f"{destination} 博物馆", f"{destination} 美术馆", f"{destination} 公园", f"{destination} 景点"]
    items: list[Attraction] = []
    seen: set[str] = set()
    for record in _nominatim_records(queries, limit=8):
        name = _record_name(record)
        if not name or name in seen or _bad_attraction_name(name) or not _looks_like_attraction_name(name):
            continue
        tags = _record_attraction_tags(record, name)
        if _is_avoided(tags, avoid):
            continue
        items.append(
            Attraction(
                name=name,
                area=destination,
                tags=tags,
                duration_minutes=_duration_for_attraction(tags),
                ticket_price=_ticket_price(tags),
                crowd_level=3,
                source="osm_nominatim",
                latitude=float(record["lat"]),
                longitude=float(record["lon"]),
            )
        )
        seen.add(name)
    return _rank_attractions(items, preferences, avoid, center_lat, center_lon)[:max_results]


def _nominatim_restaurants(
    destination: str,
    preferences: list[str],
    avoid: list[str],
    max_results: int,
    center_lat: float,
    center_lon: float,
) -> list[Restaurant]:
    records = _nominatim_records([f"{destination} 餐厅", f"{destination} 本帮菜", f"{destination} 饭店"], limit=12)
    items: list[Restaurant] = []
    seen: set[str] = set()
    for record in records:
        name = _record_name(record)
        if not name or name in seen or _bad_restaurant_name(name):
            continue
        if _is_beverage_first_venue(name, preferences):
            continue
        tags = ["food", "local"]
        if _is_avoided(tags, avoid):
            continue
        items.append(
            Restaurant(
                name=name,
                area=destination,
                tags=tags,
                average_price=_restaurant_price(tags),
                source="osm_nominatim",
                latitude=float(record["lat"]),
                longitude=float(record["lon"]),
            )
        )
        seen.add(name)
    return _rank_restaurants(items, preferences, avoid, center_lat, center_lon)[:max_results]


def _nominatim_records(queries: list[str], limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        for query in queries:
            response = client.get(NOMINATIM_URL, params={"q": query, "format": "json", "limit": limit, "accept-language": "zh-CN"})
            response.raise_for_status()
            records.extend(response.json())
    return records


def _record_name(record: dict[str, Any]) -> str:
    namedetails = record.get("namedetails") or {}
    if namedetails.get("name:zh"):
        return namedetails["name:zh"]
    display = record.get("display_name", "")
    return display.split(",")[0].strip()


def _record_attraction_tags(record: dict[str, Any], name: str) -> list[str]:
    text = f"{record.get('class', '')} {record.get('type', '')} {name}".lower()
    result: list[str] = []
    if any(word in text for word in ["museum", "博物馆", "美术馆", "gallery"]):
        result.append("museum")
    if any(word in text for word in ["art", "美术", "艺术", "gallery"]):
        result.append("art")
    if any(word in text for word in ["park", "公园", "garden"]):
        result.extend(["nature", "relaxed"])
    if not result:
        result.append("sightseeing")
    return _unique(result)


def _baidu_attraction_queries(destination: str, preferences: list[str]) -> list[str]:
    queries = [f"{destination} 景点", f"{destination} 博物馆", f"{destination} 历史景点", f"{destination} 公园"]
    if "museum" in preferences or "art" in preferences:
        queries = [f"{destination} 博物馆", f"{destination} 美术馆", *queries]
    if "history" in preferences:
        queries = [f"{destination} 纪念馆", f"{destination} 故居", *queries]
    if "nature" in preferences:
        queries = [f"{destination} 公园", f"{destination} 自然景点", *queries]
    return _unique(queries)


def _baidu_location(poi: dict[str, Any]) -> tuple[float, float] | None:
    location = poi.get("location") or {}
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        return None
    return float(lat), float(lng)


def _baidu_area(poi: dict[str, Any], fallback: str) -> str:
    return poi.get("area") or ((poi.get("detail_info") or {}).get("area")) or fallback


def _baidu_attraction_tags(poi: dict[str, Any]) -> list[str]:
    detail = poi.get("detail_info") or {}
    text = " ".join(
        str(value)
        for value in [poi.get("name", ""), poi.get("address", ""), detail.get("tag", ""), detail.get("type", "")]
        if value
    ).lower()
    result: list[str] = []
    if _has_any(text, ["博物馆", "纪念馆", "展览馆", "陈列馆"]):
        result.extend(["museum", "history"])
    if _has_any(text, ["美术馆", "艺术馆", "画廊", "艺术"]):
        result.extend(["museum", "art"])
    if _has_any(text, ["公园", "花园", "植物园", "湿地", "湖", "山", "森林"]):
        result.extend(["nature", "relaxed"])
    if _has_any(text, ["故居", "古迹", "遗址", "历史", "人文", "纪念"]):
        result.append("history")
    if _has_any(text, ["步行街", "老街", "外滩", "广场", "街区"]):
        result.extend(["citywalk", "local"])
    return _unique(result or ["sightseeing"])


def _baidu_restaurant_tags(poi: dict[str, Any]) -> list[str]:
    detail = poi.get("detail_info") or {}
    text = " ".join(
        str(value)
        for value in [poi.get("name", ""), poi.get("address", ""), detail.get("tag", ""), detail.get("type", "")]
        if value
    ).lower()
    result = ["food", "local"]
    cuisine_map = {
        "本帮": "shanghai",
        "江浙": "jiangzhe",
        "小吃": "snack",
        "面": "noodle",
        "火锅": "hotpot",
        "粤菜": "cantonese",
        "川菜": "sichuan",
        "湘菜": "hunan",
        "日料": "japanese",
        "咖啡": "coffee",
        "茶": "tea",
    }
    for word, tag in cuisine_map.items():
        if word in text:
            result.append(tag)
    if _has_any(text, CHAIN_RESTAURANT_KEYWORDS):
        result.append("chain")
    return _unique(result)


def _baidu_ticket_price(poi: dict[str, Any], tags: list[str]) -> float:
    detail = poi.get("detail_info") or {}
    for key in ["price", "ticket", "ticket_price"]:
        value = detail.get(key)
        if value not in {None, ""}:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return _ticket_price(tags)


def _baidu_restaurant_price(poi: dict[str, Any], tags: list[str]) -> float:
    detail = poi.get("detail_info") or {}
    value = detail.get("price")
    if value not in {None, ""}:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return _restaurant_price(tags)


def _baidu_crowd_level(poi: dict[str, Any], tags: list[str]) -> int:
    comments = _baidu_comment_count(poi)
    rating = _baidu_rating(poi)
    if comments >= 200 or rating >= 4.7:
        return 4
    if comments >= 60 or rating >= 4.2:
        return 3
    if "nature" in tags:
        return 2
    return _crowd_level({}, tags)


def _baidu_rating(poi: dict[str, Any]) -> float:
    detail = poi.get("detail_info") or {}
    for key in ["overall_rating", "overall_rating_star"]:
        value = detail.get(key)
        if value not in {None, ""}:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _baidu_comment_count(poi: dict[str, Any]) -> int:
    detail = poi.get("detail_info") or {}
    for key in ["comment_num", "comment_num_label"]:
        value = detail.get(key)
        if value not in {None, ""}:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return 0


def _quality_signal(rating: float, review_count: int) -> float:
    if rating <= 0:
        return 0.0
    weighted_rating = ((RATING_PRIOR_WEIGHT * RATING_PRIOR_MEAN) + (review_count * rating)) / (RATING_PRIOR_WEIGHT + max(review_count, 0))
    popularity_bonus = min(math.log1p(max(review_count, 0)), 6.0) * 0.18
    return weighted_rating + popularity_bonus


def _name(tags: dict[str, Any]) -> str:
    return tags.get("name:zh") or tags.get("name") or tags.get("name:en") or ""


def _area(tags: dict[str, Any], fallback: str) -> str:
    return tags.get("addr:district") or tags.get("addr:suburb") or tags.get("addr:city") or fallback


def _point(element: dict[str, Any]) -> tuple[float, float] | None:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def _attraction_tags(tags: dict[str, Any]) -> list[str]:
    result: list[str] = []
    tourism = tags.get("tourism", "")
    leisure = tags.get("leisure", "")
    historic = tags.get("historic")
    if tourism in {"museum", "gallery"}:
        result.extend(["museum", "art" if tourism == "gallery" else "history"])
    if tourism in {"attraction", "artwork"}:
        result.append("sightseeing")
    if historic:
        result.append("history")
    if leisure in {"park", "garden"}:
        result.extend(["nature", "relaxed"])
    if tags.get("wikidata") or tags.get("wikipedia"):
        result.append("landmark")
    return _unique(result or ["local"])


def _restaurant_tags(tags: dict[str, Any]) -> list[str]:
    amenity = tags.get("amenity", "")
    cuisine = tags.get("cuisine", "")
    name = _name(tags).lower()
    result = ["food"]
    if amenity == "cafe":
        result.extend(["coffee", "relaxed"])
    elif amenity in {"fast_food", "food_court"}:
        result.append("snack")
    else:
        result.append("local")
    if _has_any(name, CHAIN_RESTAURANT_KEYWORDS):
        result.append("chain")
    if cuisine:
        result.extend(cuisine.replace(";", ",").split(",")[:3])
    return _unique([tag.strip() for tag in result if tag.strip()])


def _ticket_price(tags: list[str]) -> float:
    if "nature" in tags:
        return 0
    if "museum" in tags:
        return 35
    if "landmark" in tags:
        return 50
    return 25


def _restaurant_price(tags: list[str]) -> float:
    if "coffee" in tags:
        return 45
    if "snack" in tags:
        return 35
    return 80


def _duration_for_attraction(tags: list[str]) -> int:
    if "museum" in tags:
        return 120
    if "nature" in tags:
        return 90
    return 80


def _crowd_level(tags: dict[str, Any], item_tags: list[str]) -> int:
    if tags.get("wikidata") or "landmark" in item_tags:
        return 4
    if "nature" in item_tags:
        return 2
    return 3


def _rank_attractions(items: list[Attraction], preferences: list[str], avoid: list[str], center_lat: float, center_lon: float) -> list[Attraction]:
    return sorted(
        items,
        key=lambda item: (
            _score(item.tags, preferences, avoid),
            _attraction_name_score(item.name),
            -item.crowd_level,
            -_distance_from_center(item.latitude, item.longitude, center_lat, center_lon),
        ),
        reverse=True,
    )


def _rank_restaurants(items: list[Restaurant], preferences: list[str], avoid: list[str], center_lat: float, center_lon: float) -> list[Restaurant]:
    return sorted(
        items,
        key=lambda item: (
            _score(item.tags, preferences, avoid),
            -10 if "chain" in item.tags else 0,
            3 if "local" in item.tags else 0,
            _restaurant_name_score(item.name),
            -_distance_from_center(item.latitude, item.longitude, center_lat, center_lon),
            -item.average_price,
        ),
        reverse=True,
    )


def _score(tags: list[str], preferences: list[str], avoid: list[str]) -> int:
    return sum(3 for tag in tags if tag in preferences) - sum(5 for tag in tags if tag in avoid)


def _is_avoided(tags: list[str], avoid: list[str]) -> bool:
    return any(tag in avoid for tag in tags)


def _bad_attraction_name(name: str) -> bool:
    lowered = name.lower()
    return _has_any(lowered, BAD_POI_KEYWORDS)


def _looks_like_attraction_name(name: str) -> bool:
    if _has_any(name, ["博物馆", "美术馆", "纪念馆", "陈列馆", "艺术馆", "公园", "花园", "豫园", "外滩", "故居", "寺", "宫", "塔", "街", "路", "广场"]):
        return True
    if any(char.isascii() and char.isalpha() for char in name) and len(name) >= 4:
        return True
    return len(name) >= 4


def _attraction_name_score(name: str) -> int:
    score = 0
    if _has_any(name, ["博物馆", "美术馆", "纪念馆", "陈列馆", "艺术馆"]):
        score += 5
    if _has_any(name, ["公园", "花园", "豫园", "外滩", "故居", "寺", "宫", "塔", "街", "路"]):
        score += 3
    if _has_any(name, ["公司", "店", "中心"]):
        score -= 4
    return score


def _bad_restaurant_name(name: str) -> bool:
    lowered = name.lower()
    return _has_any(lowered, BAD_RESTAURANT_KEYWORDS) or _has_any(lowered, CHAIN_RESTAURANT_KEYWORDS)


def _is_beverage_first_venue(name: str, preferences: list[str]) -> bool:
    if "coffee" in preferences:
        return False
    lowered = name.lower()
    return _has_any(lowered, BEVERAGE_VENUE_KEYWORDS)


def _restaurant_name_score(name: str) -> int:
    score = 0
    if any("\u4e00" <= char <= "\u9fff" for char in name):
        score += 3
    if _has_any(name, ["食", "餐", "馆", "菜", "面", "饭", "小吃"]):
        score += 3
    if name.isascii():
        score -= 2
    return score


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _distance_from_center(lat: float | None, lon: float | None, center_lat: float, center_lon: float) -> float:
    if lat is None or lon is None:
        return 999
    return _haversine_km(center_lat, center_lon, lat, lon)


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result
