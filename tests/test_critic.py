from tripmind.agents.critic import CriticAgent
from tripmind.agents.planner import PlannerAgent
from tripmind.schemas import BudgetBreakdown, DayPlan, Itinerary, ItineraryItem, TransitEstimate, TripRequest, UserMemory


def test_critic_flags_budget_exceeded() -> None:
    request = TripRequest(destination="北京", days=2, budget=100, preferences=["history"])
    itinerary = PlannerAgent().plan(request, UserMemory(user_id="u1"))
    report = CriticAgent().review(request, itinerary, UserMemory(user_id="u1"))

    assert any(issue.code == "budget_exceeded" for issue in report.issues)
    assert report.passed is False


def test_critic_reports_scattered_route_metrics() -> None:
    request = TripRequest(destination="武汉", days=1, budget=800, preferences=["museum"])
    itinerary = Itinerary(
        request=request,
        days=[
            DayPlan(
                day=1,
                theme="Scattered route",
                items=[
                    ItineraryItem(time_of_day="Morning", title="江汉关博物馆", kind="attraction", area="江汉区", duration_minutes=120, estimated_cost=35, notes="museum", latitude=30.58, longitude=114.29),
                    ItineraryItem(time_of_day="Lunch", title="汉阳面馆", kind="restaurant", area="汉阳区", duration_minutes=70, estimated_cost=40, notes="food, local", latitude=30.55, longitude=114.20),
                    ItineraryItem(time_of_day="Afternoon", title="蔡甸美术馆", kind="attraction", area="蔡甸区", duration_minutes=120, estimated_cost=0, notes="museum, art", latitude=30.56, longitude=113.98),
                    ItineraryItem(time_of_day="Dinner", title="新洲酒楼", kind="restaurant", area="新洲区", duration_minutes=70, estimated_cost=55, notes="food, local", latitude=30.87, longitude=114.67),
                ],
                transit=[
                    TransitEstimate(origin="江汉关博物馆", destination="汉阳面馆", minutes=54, cost=27.7, mode="estimate", distance_km=10.66),
                    TransitEstimate(origin="汉阳面馆", destination="蔡甸美术馆", minutes=88, cost=54.4, mode="estimate", distance_km=20.93),
                    TransitEstimate(origin="蔡甸美术馆", destination="新洲酒楼", minutes=266, cost=192.8, mode="estimate", distance_km=74.15),
                ],
                estimated_cost=378.9,
            )
        ],
        budget=BudgetBreakdown(attractions=35, restaurants=95, transit=274.9, buffer=48.6, total=453.5, currency="CNY"),
    )

    report = CriticAgent().review(request, itinerary, UserMemory(user_id="u1"))

    codes = {issue.code for issue in report.issues}
    assert "cross_area_jumps" in codes
    assert "long_route_leg" in codes
    assert report.metrics["cross_area_jump_count"] >= 3
    assert report.metrics["max_leg_km"] >= 70
