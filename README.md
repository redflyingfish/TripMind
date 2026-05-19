# TripMind

TripMind 是一个面向中文场景的智能旅行规划 Agent 应用。  
用户输入自然语言旅行需求后，系统会完成结构化解析、真实 POI 检索、行程规划、路线审查、偏好记忆，并输出可读的 Markdown 行程方案。

TripMind 当前聚焦三个问题:

- 把模糊的旅行需求转成结构化约束
- 在真实 POI 与路线数据上生成可执行的多日行程
- 在输出前对节奏、预算、偏好覆盖和路线合理性做一次审查

## Features

- `Multi-agent workflow`: `IntentAgent -> PlannerAgent -> CriticAgent -> Memory`
- `Structured outputs`: 所有核心输入输出都用 Pydantic 定义
- `Real tool layer`: LLM 不直接访问外部系统，只通过显式工具完成 POI、路线和预算相关操作
- `MCP integration`: 景点/餐厅/交通/预算工具通过 MCP 暴露给规划层
- `Real map providers`: 支持百度地图、高德地图，并在失败时回退到开放地图数据
- `Quality guardrails`: 结合评分、热度、区域聚类、跨区跳点检测和路线压缩优化行程质量
- `Stateful runtime`: 内置状态机、checkpoint、trace 和 memory 持久化
- `Application surface`: 提供 FastAPI 后端、React 前端、Typer CLI 和核心测试

## How It Works

```text
Natural-language request
  -> IntentAgent      -> TripRequest
  -> PlannerAgent     -> MCP tools + LLM itinerary draft
  -> CriticAgent      -> metrics + review issues
  -> Human confirm    -> confirmed state
  -> Memory update    -> UserMemory
  -> Markdown renderer
```

运行状态采用固定流程，而不是自由协商式多轮聊天:

`collecting -> planning -> reviewing -> awaiting_confirmation -> confirmed`

这种设计的重点是稳定、可测、可恢复，并且方便把每一步的输入、输出和中间产物都留痕。

## Architecture

### Agents

- `IntentAgent`: 解析自然语言需求，生成结构化 `TripRequest`
- `PlannerAgent`: 调用景点/餐厅/交通/预算工具，结合 LLM 生成多日行程
- `CriticAgent`: 从预算、节奏、偏好覆盖、路线跨度等角度审查方案
- `Memory`: 记录用户长期偏好，如喜欢博物馆、预算敏感、不喜欢太赶

### Typed Schemas

所有 Agent handoff 都通过 `tripmind/schemas.py` 中的 Pydantic 模型完成，例如:

- `TripRequest`
- `IntentResult`
- `Itinerary`
- `CritiqueReport`
- `WorkflowRun`

### Tool Layer

LLM 只负责“理解、规划、审查”，不直接操作外部系统。真实执行统一在工具层完成。

当前主要工具包括:

- `attractions_search`
- `restaurant_search`
- `estimate_transit`
- `estimate_budget`

对应数据来源:

- `baidu`: 百度地图 Place / Geocoding API
- `amap`: 高德地图 POI / 路径 API
- `fallback`: Nominatim + Overpass + OpenStreetMap

### Runtime Artifacts

每次运行都会保留一组可调试产物:

- `selected_attractions`
- `selected_restaurants`
- `route_segments`
- `budget_summary`
- `memory_pack`
- `evaluation_metrics`
- `data_sources`
- `data_warnings`

这些产物主要用于调试、评估和后续迭代。

## Tech Stack

- Backend: Python, FastAPI, Typer, Pydantic
- Frontend: React, Vite
- Agent runtime: typed state machine, checkpoint, trace, memory
- LLM: OpenAI-compatible API, 默认使用 DashScope / Qwen
- Tools: MCP
- Travel data: Baidu Map API / Amap API / OSM fallback

## Quick Start

### 1. Create environment

```bash
make venv
make install
make frontend-install
```

如果你不使用 `make`，也可以手动执行:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd frontend && npm install
```

### 2. Configure `.env`

先复制示例文件:

```bash
cp .env.example .env
```

最小可用配置示例:

```bash
DASHSCOPE_API_KEY=your-api-key
TRIPMIND_LLM_PROVIDER=dashscope
TRIPMIND_LLM_MODEL=qwen3.6-plus
TRIPMIND_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TRIPMIND_LLM_WIRE_API=chat
TRIPMIND_LLM_ENABLE_THINKING=false
TRIPMIND_LLM_TIMEOUT_SECONDS=45
TRIPMIND_LLM_MAX_RETRIES=1
TRIPMIND_LLM_CACHE=false
TRIPMIND_LLM_CACHE_PATH=.tripmind_llm_cache.json

TRIPMIND_TRAVEL_PROVIDER=baidu
BAIDU_MAP_AK=your-baidu-map-ak
```

如果你使用高德地图:

```bash
TRIPMIND_TRAVEL_PROVIDER=amap
AMAP_API_KEY=your-amap-web-service-key
```

说明:

- LLM 层支持任意 OpenAI-compatible 接口，不强绑定某一家厂商
- 默认推荐 `qwen3.6-plus`，因为它对中文指令和结构化输出比较稳
- 如果地图 provider 未配置或请求失败，TripMind 会自动回退到开放地图数据

### 3. Start backend and frontend

后端:

```bash
make backend
```

前端:

```bash
make frontend
```

同时启动:

```bash
make dev
```

默认地址:

- Backend: `http://127.0.0.1:8010`
- Frontend: `http://127.0.0.1:5174`

## Usage

### Web App

打开前端页面后，可以直接输入中文需求，例如:

```text
我想去上海玩 2 天，预算 1200 元，喜欢博物馆和本地美食，不要太赶。
```

前端会展示:

- 旅行需求提交
- Agent 执行结果
- 每日行程方案
- 预算估算
- 风险提示
- 调整建议
- 数据来源与部分评分/热度信息

### CLI

```bash
.venv/bin/tripmind plan "我想去上海玩2天，预算900元，喜欢博物馆和美食，不要太赶" --user-id demo
```

查看 memory:

```bash
.venv/bin/tripmind memory --user-id demo
```

无真实 LLM 的本地演示模式:

```bash
.venv/bin/tripmind plan "我想去上海玩2天，预算900元，喜欢博物馆和美食，不要太赶" --mock
```

### HTTP API

健康检查:

```bash
curl http://127.0.0.1:8010/health
```

生成旅行方案:

```bash
curl -X POST http://127.0.0.1:8010/trips/plan \
  -H "Content-Type: application/json" \
  -d '{
    "text": "去杭州2天，预算900元，喜欢博物馆和自然，轻松一点",
    "user_id": "demo"
  }'
```

只跑到审查阶段，等待人工确认:

```bash
curl -X POST http://127.0.0.1:8010/trips/plan \
  -H "Content-Type: application/json" \
  -d '{
    "text": "去成都玩2天，预算1000元，想吃好一点",
    "user_id": "demo",
    "auto_confirm": false
  }'
```

确认一个已有 run:

```bash
curl -X POST http://127.0.0.1:8010/runs/<run_id>/confirm
```

## Example Output

TripMind 会输出一份 Markdown 行程单，内容包括:

- 每日安排
- 预算估算
- 风险提醒
- 调整建议
- 关键评估指标
- Agent workflow trace

示例片段:

```markdown
# TripMind Itinerary: 上海

- State: `confirmed`
- Duration: 2 day(s)
- Pace: relaxed
- Budget: 900 CNY

## Daily Plan

### Day 1: Museum and food
- **Morning** · 上海博物馆 (attraction, 人民广场, 120 min, ~0 CNY, source: baidu, rating: 4.7, reviews: 12456)
- **Lunch** · 老字号本帮菜馆 (restaurant, 黄浦, 75 min, ~88 CNY, source: baidu, rating: 4.5, reviews: 6321)
```

## Project Structure

```text
tripmind/
  agents/          Intent / Planner / Critic
  api.py           FastAPI app
  cli.py           Typer CLI
  llm.py           OpenAI-compatible LLM wrapper
  mcp_client.py    MCP client
  mcp_server.py    MCP tool server
  memory.py        User preference storage
  renderer.py      Markdown renderer
  runtime.py       Stateful workflow runtime
  schemas.py       Pydantic schemas
  travel_data.py   Map provider + ranking + fallback logic
frontend/
  src/             React UI
tests/             Core test cases
```

## Evaluation and Reliability

TripMind 不追求“看起来聪明”，而是尽量把效果问题拆成可观测指标和可落地机制。

当前已实现的工程约束包括:

- 结构化输入输出，减少 agent 自由发挥
- checkpoint 持久化，便于中断恢复
- trace 记录每个 agent 的摘要、耗时和工具调用
- memory 按 `user_id` 隔离
- cache 支持重复请求复用 LLM 结果
- provider warning 回退提示，便于判断真实数据是否生效

Critic 会重点检查:

- 预算是否超出
- 节奏是否过赶
- 偏好是否覆盖
- 是否出现重复点位
- 是否存在跨区跳点和过长路段

## Tests

运行测试:

```bash
make test
```

当前核心测试覆盖:

- intent parsing
- runtime state transition
- memory isolation
- planner route compaction
- critic metrics and issue detection
- LLM trace and cache behavior
- provider data ranking behavior

## Scope and Limitations

当前版本刻意保持“小而完整”，以下能力暂不纳入:

- 酒店、机票、门票预订
- 实时营业时间与闭园信息校验
- 用户登录、多用户协作、支付系统
- 完整 OTA 交易闭环

TripMind 当前更关注规划质量、流程稳定性和工具边界，而不是过早扩成一个交易型旅游平台。

## Roadmap

- 接入更多国内旅游数据源，增强评分与热度信号
- 增加 rerank / regenerate / compare-plan 工作流
- 增加更细粒度的 HITL 节点和评估面板
- 增加部署配置与在线演示版本

## Development Notes

- 请不要提交你的真实 `.env`、地图 API Key 或其他私密凭据
- 代码注释以英文为主，项目说明和迭代记录可以使用中文
- 如果你发现 provider 返回结果不稳定，优先查看输出里的 `data_sources` 和 `data_warnings`
