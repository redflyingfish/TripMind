from __future__ import annotations

import re
from datetime import date

from tripmind.llm import OpenAIModelClient
from tripmind.memory import infer_memory_preferences
from tripmind.schemas import IntentResult, Pace, TripRequest, UserMemory


CITY_NAMES = ["上海", "北京", "杭州", "成都", "东京", "京都", "巴黎", "广州", "深圳", "南京", "苏州"]
CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
TIME_SLOT_KEYWORDS = {
    "Breakfast": ["早餐", "早饭", "早点", "早午餐", "brunch"],
    "Morning": ["上午", "早上", "晨间"],
    "Lunch": ["午饭", "午餐", "中饭", "中餐"],
    "Afternoon": ["下午", "午后"],
    "Dinner": ["晚饭", "晚餐"],
    "Evening": ["晚上", "夜间", "夜游"],
}


class IntentAgent:
    """Parse natural language into a structured TripRequest."""

    def __init__(self, llm: OpenAIModelClient | None = None) -> None:
        self.llm = llm

    def parse(self, text: str, memory: UserMemory) -> IntentResult:
        if self.llm:
            return self._parse_with_llm(text, memory)
        return self._parse_deterministic(text, memory)

    def _parse_with_llm(self, text: str, memory: UserMemory) -> IntentResult:
        system = _intent_system_prompt()
        user = (
            f"用户 memory:\n{memory.model_dump_json(exclude={'updated_at'}, exclude_none=True)}\n\n"
            f"用户原始需求:\n{text}\n\n"
            "请返回 IntentResult。若 request 不为 null，必须把原始需求原样放入 request.raw_text。"
        )
        result = self.llm.parse(system=system, user=user, schema=IntentResult)
        if result.request:
            result.request.raw_text = text
            result.request.preferences = _unique([*memory.likes, *result.request.preferences])
            result.request.avoid = _unique([*memory.dislikes, *result.request.avoid])
        _post_process_intent_result(result, text, memory)
        return result

    def _parse_deterministic(self, text: str, memory: UserMemory) -> IntentResult:
        destination = self._parse_destination(text)
        missing_fields: list[str] = []
        blocking_missing_fields: list[str] = []
        branchable_missing_fields: list[str] = []
        advisory_missing_fields: list[str] = []
        notes: list[str] = []
        clarification_questions: list[str] = []

        if not destination:
            missing_fields.append("destination")
            blocking_missing_fields.append("destination")
            clarification_questions.append("你想去哪个城市或具体区域？")
            notes.append("Please provide a destination city.")

        days = self._parse_days(text) or 2
        budget, currency = self._parse_budget(text)
        start_date = self._parse_date(text)
        likes, dislikes, pace_hint, budget_sensitive = infer_memory_preferences(text)
        pace = pace_hint or memory.preferred_pace or Pace.balanced
        requested_time_slots, avoid_time_slots = self._parse_time_slot_preferences(text, pace)

        if budget is None:
            missing_fields.append("budget")
            branchable_missing_fields.append("budget")
            notes.append("Budget is missing; runtime may generate multiple budget tiers.")
            if memory.budget_sensitive:
                notes.append("User memory indicates budget sensitivity; planner will prefer lower-cost choices.")
        if start_date is None:
            missing_fields.append("start_date")
            advisory_missing_fields.append("start_date")
            notes.append("No start date was provided; opening hours and date-specific constraints are not checked.")

        if not destination:
            return IntentResult(
                request=None,
                missing_fields=missing_fields,
                blocking_missing_fields=blocking_missing_fields,
                branchable_missing_fields=branchable_missing_fields,
                advisory_missing_fields=advisory_missing_fields,
                clarification_questions=clarification_questions,
                confidence=0.55,
                notes=notes,
            )

        request = TripRequest(
            destination=destination,
            start_date=start_date,
            days=days,
            budget=budget,
            currency=currency,
            preferences=_unique([*memory.likes, *likes]),
            avoid=_unique([*memory.dislikes, *dislikes]),
            requested_time_slots=requested_time_slots,
            avoid_time_slots=avoid_time_slots,
            pace=pace,
            raw_text=text,
        )
        confidence = 0.9 if budget is not None or likes else 0.78
        if budget_sensitive:
            notes.append("Detected budget-sensitive language.")
        return IntentResult(
            request=request,
            missing_fields=missing_fields,
            blocking_missing_fields=blocking_missing_fields,
            branchable_missing_fields=branchable_missing_fields,
            advisory_missing_fields=advisory_missing_fields,
            clarification_questions=clarification_questions,
            confidence=confidence,
            notes=notes,
        )

    def _parse_destination(self, text: str) -> str | None:
        for city in CITY_NAMES:
            if city in text:
                return city
        match = re.search(r"(?:去|到|游玩|旅行到)([\u4e00-\u9fa5A-Za-z]{2,20})", text)
        if match:
            return match.group(1).strip("，。,. ")
        return None

    def _parse_days(self, text: str) -> int | None:
        match = re.search(r"(\d{1,2})\s*(?:天|日|days?)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"([一二两三四五六七八九十])\s*(?:天|日)", text)
        if match:
            return CHINESE_DIGITS[match.group(1)]
        return None

    def _parse_budget(self, text: str) -> tuple[float | None, str]:
        match = re.search(r"(?:预算|budget)[^\d]*(\d+(?:\.\d+)?)\s*(元|人民币|rmb|RMB|块|美元|usd|USD)?", text)
        if not match:
            match = re.search(r"(\d+(?:\.\d+)?)\s*(元|人民币|rmb|RMB|块|美元|usd|USD)", text)
        if not match:
            return None, "CNY"
        amount = float(match.group(1))
        unit = match.group(2) or "元"
        currency = "USD" if unit.lower() in {"usd", "美元"} else "CNY"
        return amount, currency

    def _parse_date(self, text: str) -> date | None:
        iso = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text)
        if iso:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        return None

    def _parse_time_slot_preferences(self, text: str, pace: Pace) -> tuple[list[str], list[str]]:
        requested: list[str] = []
        avoid: list[str] = []
        for slot, keywords in TIME_SLOT_KEYWORDS.items():
            if any(f"只要{word}" in text or f"只安排{word}" in text or f"想要{word}" in text for word in keywords):
                requested.append(slot)
            if any(f"不要{word}" in text or f"不想要{word}" in text or f"不安排{word}" in text for word in keywords):
                avoid.append(slot)

        if ("只要" in text or "只安排" in text) and "和" in text:
            for slot, keywords in TIME_SLOT_KEYWORDS.items():
                if any(word in text for word in keywords):
                    requested.append(slot)

        if not requested:
            if any(phrase in text for phrase in ["只给我下午和晚上", "下午和晚上就行", "下午晚上即可"]):
                requested.extend(["Afternoon", "Dinner", "Evening"])
            elif any(phrase in text for phrase in ["只给我下午", "只安排下午", "下午就行"]):
                requested.extend(["Afternoon", "Dinner"])
            elif any(phrase in text for phrase in ["只给我晚上", "晚上就行", "晚点出门"]):
                requested.extend(["Dinner", "Evening"])
            elif any(phrase in text for phrase in ["只给我早餐", "只安排早餐", "早餐推荐", "想吃早餐", "早饭推荐"]):
                requested.extend(["Breakfast"])

        if "不要上午行程" in text or "不用上午行程" in text or "不想早起" in text:
            avoid.append("Morning")
        if pace == Pace.relaxed and ("睡到自然醒" in text or "中午再开始" in text or "下午再开始" in text):
            avoid.append("Morning")
            requested.extend(["Afternoon", "Dinner"])

        requested = _unique([slot for slot in requested if slot not in avoid])
        avoid = _unique(avoid)
        return requested, avoid


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _post_process_intent_result(result: IntentResult, text: str, memory: UserMemory) -> None:
    missing = set(result.missing_fields)
    blocking = set(result.blocking_missing_fields)
    branchable = set(result.branchable_missing_fields)
    advisory = set(result.advisory_missing_fields)

    if result.request is None:
        blocking.add("destination")
        missing.add("destination")

    if result.request is not None and result.request.budget is None:
        missing.add("budget")
        branchable.add("budget")

    if result.request is not None and result.request.start_date is None:
        missing.add("start_date")
        advisory.add("start_date")

    if "destination" in missing and not result.clarification_questions:
        result.clarification_questions.append("你想去哪个城市或具体区域？")

    if "budget" in branchable and memory.budget_sensitive:
        result.notes.append("User memory indicates budget sensitivity; planner should prioritize cheaper variants.")

    result.missing_fields = _unique(list(missing))
    result.blocking_missing_fields = _unique(list(blocking))
    result.branchable_missing_fields = _unique(list(branchable))
    result.advisory_missing_fields = _unique(list(advisory))
    result.clarification_questions = _unique(result.clarification_questions)
    result.notes = _unique(result.notes)


def _intent_system_prompt() -> str:
    return """
你是 TripMind 的 IntentAgent，也是一个专业的旅行需求理解助手。你只负责理解用户需求，并把它整理成结构化结果。

你不负责：
- 规划每天行程
- 推荐具体景点或餐厅
- 调用工具
- 编造用户没有明确表达的信息

下面出现的英文词，例如 `request`、`destination`、`preferences`，都只是返回结果里的字段名。你可以把它们理解成固定表单键名，不需要向用户解释这些英文。

你的输出固定分成 4 个部分：

1. `request`
这是整理后的旅行需求对象。只有在关键信息足够时才填写。

`request` 中各字段的含义：
- `destination`：目的地，通常是城市、城区或明确区域
- `start_date`：出发日期；只有用户明确说了具体日期才填
- `days`：游玩天数
- `budget`：预算金额。默认理解为本次游玩预算，不含机票、火车票、酒店，除非用户明确说明
- `currency`：货币，默认 `CNY`
- `travelers`：同行人数；用户没说时默认 1
- `preferences`：用户偏好的内容，使用少量英文短标签
- `avoid`：用户不喜欢、想避开的内容，也使用少量英文短标签
- `pace`：行程节奏，只能填 `relaxed`、`balanced`、`packed`
- `raw_text`：原始用户输入，必须原样保留

2. `missing_fields`
这是还缺少的关键信息列表。
如果缺少目的地，必须把 `"destination"` 放进去。

3. `confidence`
表示你对本次理解结果的把握程度，范围 0 到 1。

4. `notes`
这是给后续 agent 的补充说明列表。
这里只写“有帮助，但不适合直接硬写进 request 的信息”。

`notes` 适合写的内容：
- 用户表达模糊
- 某个判断带有推断成分
- 用户疑似预算敏感，但没给出明确金额
- 某些中文偏好已被你映射成内部标签
- 后续规划阶段值得继续关注的限制条件

你要产出的结构可以理解成这样：
{
  "request": {
    "destination": "杭州",
    "start_date": null,
    "days": 3,
    "budget": 1200,
    "currency": "CNY",
    "travelers": 2,
    "preferences": ["museum", "food"],
    "avoid": ["crowded"],
    "pace": "relaxed",
    "raw_text": "五一后想去杭州玩三天，两个人，预算1200，想轻松一点，多吃点本地馆子"
  },
  "missing_fields": [],
  "confidence": 0.86,
  "notes": ["用户语气显示偏预算敏感"]
}

关键规则：
- 如果目的地缺失，`request` 必须是 `null`
- 不要编造目的地、日期、预算、人数
- 不确定的信息不要硬填进 `request`，改写进 `notes`
- 输出必须是结构化结果，不要附加额外解释
- `raw_text` 必须保留用户原始输入

中文语义理解规则：
- 你需要能够理解用户的含义，解析出其中的意思和我们所需的字段相关行信息，解析成字段里的信息。
- “不要太赶 / 松弛 / 轻松一点 / 慢慢逛 / 别排太满”通常表示 `pace = relaxed`
- “紧凑一点 / 多安排几个 / 特种兵 / 能多玩就多玩”通常表示 `pace = packed`
- 没有明显倾向时，使用 `balanced`


偏好和避雷点处理规则：
- 用户大概率会用中文描述偏好，你先准确理解中文原意，再映射成少量稳定标签
- `preferences` 和 `avoid` 最终使用英文短标签，方便后续检索和规划
- 标签宁可少而准，不要为了凑数量过度概括
- 某些细微语气、模糊限制、软偏好，如果不适合强行转成标签，可以写进 `notes`

常用标签参考：
- `museum`：博物馆、展览、纪念馆、美术馆
- `history`：历史文化、古迹、故居、人文体验
- `food`：美食、小吃、吃吃喝喝
- `nature`：自然风景、公园、湖边、山水、绿地
- `art`：艺术、画廊、当代艺术、审美体验
- `coffee`：咖啡、咖啡馆
- `citywalk`：citywalk、散步、压马路、街区闲逛
- `local`：本地人常去、烟火气、在地体验、老字号
- `photography`：拍照、出片、打卡
- `shopping`：购物、逛商场、买东西
- `tea`：喝茶、茶馆、茶体验
- `architecture`：建筑、老建筑、特色街区建筑
- `crowded`：人多、排队、拥挤、太热门
- `expensive`：太贵、消费高、性价比低
- `packed`：太赶、塞太满、强度过高

理解示例：
- “想看博物馆和老建筑，不想太赶”
  可以理解为：`preferences` 包含 `museum`, `architecture`；`pace = relaxed`
- “想吃点本地小吃，别去太网红太排队的地方”
  可以理解为：`preferences` 包含 `food`, `local`；`avoid` 可包含 `crowded`
- “预算卡得比较死，但也想舒服一点”
  如果没有明确金额，不要硬填 `budget`；可以在 `notes` 里提示预算敏感
- “带爸妈去逛逛，别太累”
  可以理解为节奏偏 `relaxed`；如果用户没明确人数，不要擅自填写 `travelers = 3`
""".strip()
