from tripmind.runtime import TripMindRuntime
from tripmind.llm import OpenAIModelClient
from tripmind.schemas import (
    CritiqueReport,
    DayPlanDraft,
    IntentResult,
    ItineraryDraft,
    Pace,
    PlannedStop,
    TripRequest,
)


class FakeLLMClient:
    model = "fake-real-llm"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.draft_calls = 0

    def parse(self, *, system: str, user: str, schema):
        self.calls.append(schema.__name__)
        if schema is IntentResult:
            return IntentResult(
                request=TripRequest(destination="上海", days=1, budget=500, preferences=["museum"], pace=Pace.relaxed, raw_text="去上海"),
                confidence=0.99,
            )
        if schema is ItineraryDraft:
            self.draft_calls += 1
            if self.draft_calls == 2:
                return ItineraryDraft(
                    days=[
                        DayPlanDraft(
                            day=1,
                            theme="Compact museum route",
                            stops=[
                                PlannedStop(time_of_day="morning", name="上海博物馆", kind="attraction"),
                                PlannedStop(time_of_day="lunch", name="大壶春", kind="restaurant"),
                                PlannedStop(time_of_day="afternoon", name="中华艺术宫", kind="attraction"),
                                PlannedStop(time_of_day="dinner", name="大壶春", kind="restaurant"),
                            ],
                        )
                    ],
                    assumptions=["LLM refined the route after critique."],
                )
            return ItineraryDraft(
                days=[
                    DayPlanDraft(
                        day=1,
                        theme="Museum and local food",
                        stops=[
                            PlannedStop(time_of_day="morning", name="中华艺术宫", kind="attraction"),
                            PlannedStop(time_of_day="lunch", name="大壶春", kind="restaurant"),
                            PlannedStop(time_of_day="afternoon", name="上海博物馆", kind="attraction"),
                            PlannedStop(time_of_day="evening", name="上海自然博物馆", kind="attraction"),
                            PlannedStop(time_of_day="dinner", name="阿娘面馆", kind="restaurant"),
                        ],
                    )
                ],
                assumptions=["LLM draft created from tool candidates."],
            )
        if schema is CritiqueReport:
            if self.draft_calls == 1:
                return CritiqueReport(
                    issues=[],
                    metrics={"cross_area_jump_count": 3, "dominant_area_ratio": 0.3},
                    passed=True,
                )
            return CritiqueReport(issues=[], metrics={"cross_area_jump_count": 0, "dominant_area_ratio": 0.8}, passed=True)
        raise AssertionError(f"Unexpected schema: {schema}")


def test_runtime_calls_llm_for_intent_planning_and_critique(tmp_path) -> None:
    fake_llm = FakeLLMClient()
    runtime = TripMindRuntime(memory_path=tmp_path / "memory.json", llm_client=fake_llm)

    run = runtime.run("去上海1天，预算500，喜欢博物馆", user_id="alice")

    assert fake_llm.calls == ["IntentResult", "ItineraryDraft", "CritiqueReport", "ItineraryDraft", "CritiqueReport"]
    assert run.itinerary is not None
    assert run.itinerary.days[0].theme == "Compact museum route"
    assert [item.time_of_day for item in run.itinerary.days[0].items] == ["Morning", "Lunch", "Afternoon", "Dinner"]
    assert run.trace[0].metadata["llm_model"] == "fake-real-llm"
    assert isinstance(run.trace[0].metadata["elapsed_ms"], int)
    assert run.artifacts.replan_count == 1
    assert "replan_directive" in run.trace[1].metadata


def test_openai_compatible_client_caches_structured_outputs(tmp_path) -> None:
    client = OpenAIModelClient(api_key="dummy", model="qwen3.6-plus")
    client.cache_path = tmp_path / "llm-cache.json"
    client.cache_enabled = True
    calls = {"count": 0}

    def fake_parse_chat(*, system, user, schema):
        calls["count"] += 1
        return IntentResult(
            request=TripRequest(destination="上海", days=1, preferences=["museum"], raw_text="去上海"),
            confidence=1.0,
        )

    client._parse_chat = fake_parse_chat

    first = client.parse(system="sys", user="user", schema=IntentResult)
    second = client.parse(system="sys", user="user", schema=IntentResult)

    assert calls["count"] == 1
    assert first == second
    assert client.last_call_info["cache_hit"] is True
