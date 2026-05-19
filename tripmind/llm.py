from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from dotenv import load_dotenv


load_dotenv()


T = TypeVar("T", bound=BaseModel)


DEFAULT_PROVIDER = "dashscope"
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_WIRE_API = "chat"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_ENABLE_THINKING = False


class LLMConfigurationError(RuntimeError):
    """Raised when the real LLM path is requested but not configured."""


class OpenAIModelClient:
    """Small wrapper for OpenAI-compatible LLM endpoints."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        wire_api: str | None = None,
    ) -> None:
        self.provider = os.getenv("TRIPMIND_LLM_PROVIDER", DEFAULT_PROVIDER)
        self.model = model or os.getenv("TRIPMIND_LLM_MODEL") or DEFAULT_MODEL
        self.api_key = api_key or _read_api_key()
        self.base_url = base_url or os.getenv("TRIPMIND_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.wire_api = (wire_api or os.getenv("TRIPMIND_LLM_WIRE_API") or DEFAULT_WIRE_API).lower()
        self.timeout = float(os.getenv("TRIPMIND_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        self.max_retries = int(os.getenv("TRIPMIND_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        self.enable_thinking = _env_bool("TRIPMIND_LLM_ENABLE_THINKING", DEFAULT_ENABLE_THINKING)
        self.cache_enabled = os.getenv("TRIPMIND_LLM_CACHE", "false").lower() in {"1", "true", "yes"}
        self.cache_path = Path(os.getenv("TRIPMIND_LLM_CACHE_PATH", ".tripmind_llm_cache.json"))
        self.last_call_info: dict[str, str | int | bool | None] = {}
        if not self.api_key:
            raise LLMConfigurationError(
                "A model API key is required for TripMind's real LLM path. "
                "Set TRIPMIND_LLM_API_KEY, OPENAI_API_KEY, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, or MOONSHOT_API_KEY. "
                "Set it in the environment, or run with --mock for local deterministic mode."
            )
        if self.wire_api not in {"chat", "responses"}:
            raise LLMConfigurationError("TRIPMIND_LLM_WIRE_API must be either 'chat' or 'responses'.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The openai package is required for real LLM calls. Install with `pip install -e .`."
            ) from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        cache_key = self._cache_key(system=system, user=user, schema=schema)
        if self.cache_enabled:
            cached = self._read_cache().get(cache_key)
            if cached is not None:
                self.last_call_info = {"schema": schema.__name__, "cache_hit": True, "wire_api": self.wire_api}
                return schema.model_validate(cached)

        if self.wire_api == "responses":
            parsed = self._parse_responses(system=system, user=user, schema=schema)
        else:
            parsed = self._parse_chat(system=system, user=user, schema=schema)

        if self.cache_enabled:
            cache = self._read_cache()
            cache[cache_key] = parsed.model_dump(mode="json")
            self._write_cache(cache)
        self.last_call_info = {"schema": schema.__name__, "cache_hit": False, "wire_api": self.wire_api}
        return parsed

    def metadata(self) -> dict[str, str | None]:
        return {
            "llm_provider": self.provider,
            "llm_model": self.model,
            "llm_base_url": self.base_url,
            "llm_wire_api": self.wire_api,
            "llm_timeout_seconds": str(self.timeout),
            "llm_enable_thinking": str(self.enable_thinking).lower(),
            "llm_cache": str(self.cache_enabled).lower(),
        }

    def _cache_key(self, *, system: str, user: str, schema: type[BaseModel]) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "wire_api": self.wire_api,
            "schema": schema.__name__,
            "system": system,
            "user": user,
        }
        return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _read_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        raw = self.cache_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _write_cache(self, cache: dict[str, dict]) -> None:
        self.cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _parse_responses(self, *, system: str, user: str, schema: type[T]) -> T:
        response = self._client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI response did not include parsed structured output.")
        return parsed

    def _parse_chat(self, *, system: str, user: str, schema: type[T]) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system}\n\n"
                        "Return only valid JSON that conforms to this JSON Schema. "
                        "Do not wrap it in Markdown fences.\n"
                        f"{schema_json}"
                    ),
                },
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": self.enable_thinking},
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("LLM chat completion did not return content.")
        return schema.model_validate_json(_extract_json(content))


def _read_api_key() -> str | None:
    for name in [
        "TRIPMIND_LLM_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
    ]:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if match:
        return match.group(1)
    raise RuntimeError("LLM response did not contain a JSON object.")
