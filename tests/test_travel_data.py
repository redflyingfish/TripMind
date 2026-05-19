import tripmind.travel_data as travel_data


def test_baidu_restaurant_search_filters_beverage_first_venues(monkeypatch) -> None:
    monkeypatch.setattr(travel_data, "geocode_place", lambda place: (31.23, 121.47))
    monkeypatch.setattr(travel_data, "_baidu_enrich_results", lambda results: results)
    monkeypatch.setattr(
        travel_data,
        "_baidu_region_search",
        lambda **kwargs: [
            {
                "uid": "1",
                "name": "禅石咖啡酒吧餐厅",
                "area": "黄浦区",
                "location": {"lat": 31.231, "lng": 121.471},
                "detail_info": {"overall_rating": 4.9, "comment_num": 200, "price": 85, "tag": "咖啡厅"},
            },
            {
                "uid": "2",
                "name": "阿娘面馆",
                "area": "黄浦区",
                "location": {"lat": 31.232, "lng": 121.472},
                "detail_info": {"overall_rating": 4.5, "comment_num": 120, "price": 42, "tag": "小吃"},
            },
        ],
    )

    results = travel_data._baidu_restaurant_search("上海", preferences=["food"], avoid=[], max_results=5)

    assert [item.name for item in results] == ["阿娘面馆"]


def test_baidu_restaurant_search_keeps_coffee_when_user_prefers_it(monkeypatch) -> None:
    monkeypatch.setattr(travel_data, "geocode_place", lambda place: (31.23, 121.47))
    monkeypatch.setattr(travel_data, "_baidu_enrich_results", lambda results: results)
    monkeypatch.setattr(
        travel_data,
        "_baidu_region_search",
        lambda **kwargs: [
            {
                "uid": "1",
                "name": "禅石咖啡酒吧餐厅",
                "area": "黄浦区",
                "location": {"lat": 31.231, "lng": 121.471},
                "detail_info": {"overall_rating": 4.6, "comment_num": 88, "price": 65, "tag": "咖啡厅"},
            }
        ],
    )

    results = travel_data._baidu_restaurant_search("上海", preferences=["food", "coffee"], avoid=[], max_results=5)

    assert [item.name for item in results] == ["禅石咖啡酒吧餐厅"]
