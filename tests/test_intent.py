from tripmind.agents.intent import IntentAgent
from tripmind.schemas import Pace, UserMemory


def test_intent_parses_chinese_trip_request() -> None:
    text = "我想去上海玩3天，预算800元，喜欢博物馆和美食，不要太赶"
    result = IntentAgent().parse(text, UserMemory(user_id="u1"))

    assert result.request is not None
    assert result.request.destination == "上海"
    assert result.request.days == 3
    assert result.request.budget == 800
    assert result.request.pace == Pace.relaxed
    assert "museum" in result.request.preferences
    assert "food" in result.request.preferences


def test_intent_requires_destination() -> None:
    result = IntentAgent().parse("周末两天，预算500，喜欢美食", UserMemory(user_id="u1"))

    assert result.request is None
    assert "destination" in result.missing_fields
    assert "destination" in result.blocking_missing_fields
    assert result.clarification_questions


def test_intent_marks_budget_as_branchable_when_missing() -> None:
    result = IntentAgent().parse("去杭州2天，喜欢博物馆和自然，轻松一点", UserMemory(user_id="u1"))

    assert result.request is not None
    assert "budget" in result.missing_fields
    assert "budget" in result.branchable_missing_fields
    assert result.needs_clarification is False


def test_intent_parses_requested_and_avoided_time_slots() -> None:
    result = IntentAgent().parse("去上海1天，只要下午和晚上，不要上午行程", UserMemory(user_id="u1"))

    assert result.request is not None
    assert "Afternoon" in result.request.requested_time_slots
    assert "Evening" in result.request.requested_time_slots
    assert "Morning" in result.request.avoid_time_slots


def test_intent_parses_breakfast_request() -> None:
    result = IntentAgent().parse("去上海1天，想要早餐推荐，再给我下午和晚上行程", UserMemory(user_id="u1"))

    assert result.request is not None
    assert "Breakfast" in result.request.requested_time_slots
