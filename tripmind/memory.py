from __future__ import annotations

import json
from pathlib import Path

from tripmind.schemas import EpisodicMemoryEntry, Pace, TripRequest, UserMemory, WorkflowRun


class JsonMemoryStore:
    """Small file-backed memory store with user_id isolation."""

    def __init__(self, path: str | Path = ".tripmind_memory.json") -> None:
        self.path = Path(path)

    def get(self, user_id: str) -> UserMemory:
        data = self._read()
        profiles = data.get("profiles", {})
        if user_id not in profiles:
            return UserMemory(user_id=user_id)
        return UserMemory.model_validate(profiles[user_id])

    def save(self, memory: UserMemory) -> None:
        data = self._read()
        profiles = data.setdefault("profiles", {})
        episodic = data.setdefault("episodic", {})
        profiles[memory.user_id] = memory.model_dump(mode="json")
        data["profiles"] = profiles
        data["episodic"] = episodic
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_from_request(self, user_id: str, request: TripRequest) -> UserMemory:
        memory = self.get(user_id)
        memory.likes = _merge(memory.likes, request.preferences)
        memory.dislikes = _merge(memory.dislikes, request.avoid)
        memory.preferred_pace = request.pace
        if request.budget is not None:
            per_day = request.budget / max(request.days, 1)
            memory.budget_sensitive = memory.budget_sensitive or per_day <= 450
        self.save(memory)
        return memory

    def add_episode(self, run: WorkflowRun) -> None:
        if run.intent.request is None or run.itinerary is None:
            return
        data = self._read()
        profiles = data.setdefault("profiles", {})
        episodic = data.setdefault("episodic", {})
        entries = episodic.setdefault(run.user_id, [])
        itinerary = run.itinerary
        entry = EpisodicMemoryEntry(
            destination=run.intent.request.destination,
            summary=_summarize_run(run),
            days=run.intent.request.days,
            budget=run.intent.request.budget,
            pace=run.intent.request.pace,
            preferences=run.intent.request.preferences,
            selected_titles=[item.title for day in itinerary.days for item in day.items],
            selected_areas=_unique([item.area for day in itinerary.days for item in day.items]),
            issue_codes=[issue.code for issue in run.critique.issues] if run.critique else [],
        )
        entries.append(entry.model_dump(mode="json"))
        episodic[run.user_id] = entries[-12:]
        data["profiles"] = profiles
        data["episodic"] = episodic
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def retrieve_episodes(
        self,
        user_id: str,
        destination: str,
        preferences: list[str],
        limit: int = 3,
    ) -> list[EpisodicMemoryEntry]:
        data = self._read()
        episodic = data.get("episodic", {})
        raw_entries = episodic.get(user_id, [])
        entries = [EpisodicMemoryEntry.model_validate(item) for item in raw_entries]
        ranked = sorted(
            entries,
            key=lambda entry: _episode_score(entry, destination, preferences),
            reverse=True,
        )
        return [entry for entry in ranked[:limit] if _episode_score(entry, destination, preferences) > 0]

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if "profiles" not in data:
            # Backward-compatible upgrade path from the old flat profile-only format.
            return {"profiles": data, "episodic": {}}
        data.setdefault("episodic", {})
        return data


def _merge(existing: list[str], incoming: list[str]) -> list[str]:
    seen = set(existing)
    result = list(existing)
    for item in incoming:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _episode_score(entry: EpisodicMemoryEntry, destination: str, preferences: list[str]) -> float:
    score = 0.0
    if entry.destination == destination:
        score += 4.0
    overlap = len(set(entry.preferences) & set(preferences))
    score += overlap * 1.5
    if entry.pace is not None:
        score += 0.1
    score -= len(entry.issue_codes) * 0.3
    return score


def _summarize_run(run: WorkflowRun) -> str:
    itinerary = run.itinerary
    request = run.intent.request
    if itinerary is None or request is None:
        return ""
    titles = [item.title for day in itinerary.days for item in day.items[:2]]
    return (
        f"{request.destination} {request.days} day plan with {', '.join(request.preferences) or 'general'} focus; "
        f"selected highlights: {', '.join(titles[:4])}."
    )


def infer_memory_preferences(text: str) -> tuple[list[str], list[str], Pace | None, bool]:
    likes: list[str] = []
    dislikes: list[str] = []
    pace: Pace | None = None
    budget_sensitive = False

    keyword_likes = {
        "博物馆": "museum",
        "美术馆": "art",
        "艺术": "art",
        "历史": "history",
        "咖啡": "coffee",
        "美食": "food",
        "本地": "local",
        "自然": "nature",
        "公园": "nature",
        "citywalk": "citywalk",
        "Citywalk": "citywalk",
    }
    keyword_dislikes = {
        "不喜欢排队": "crowded",
        "讨厌排队": "crowded",
        "不想购物": "shopping",
        "不爱购物": "shopping",
        "不喜欢太赶": "packed",
        "不要太赶": "packed",
    }
    for word, tag in keyword_likes.items():
        if word in text:
            likes.append(tag)
    for word, tag in keyword_dislikes.items():
        if word in text:
            dislikes.append(tag)
    if any(word in text for word in ["不赶", "不要太赶", "别太赶", "轻松", "松弛", "慢节奏"]):
        pace = Pace.relaxed
    elif any(word in text for word in ["紧凑", "多安排", "多逛", "特种兵"]):
        pace = Pace.packed
    if any(word in text for word in ["预算敏感", "省钱", "便宜", "控制预算", "穷游"]):
        budget_sensitive = True

    return likes, dislikes, pace, budget_sensitive
