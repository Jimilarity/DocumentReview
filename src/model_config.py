import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from agent_trace import AGENT_TRACE_CALLBACK


_MODEL_RATE_LIMITER: InMemoryRateLimiter | None = None
_MODEL_MAX_RETRIES = 8
_MODEL_TIMEOUT_SECONDS = 120
_VISION_MODEL_TIMEOUT_SECONDS = 360


def _get_model_rate_limiter() -> InMemoryRateLimiter | None:
    """返回共享限速器；RPS=0 时沿用 ChatOpenAI 的默认无限速行为。"""

    global _MODEL_RATE_LIMITER
    requests_per_second = float(
        os.getenv("REVIEW_MODEL_REQUESTS_PER_SECOND", "0")
    )
    if requests_per_second == 0:
        return None
    if _MODEL_RATE_LIMITER is None:
        _MODEL_RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
        )
    return _MODEL_RATE_LIMITER


def _transport_options(
    *,
    timeout_seconds: int = _MODEL_TIMEOUT_SECONDS,
    max_retries: int = _MODEL_MAX_RETRIES,
) -> dict:
    options = {
        "max_retries": max_retries,
        "timeout": timeout_seconds,
    }
    rate_limiter = _get_model_rate_limiter()
    if rate_limiter is not None:
        options["rate_limiter"] = rate_limiter
    return options


def _get_required_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable, tried: {', '.join(names)}")


def build_vision_model(
    *,
    enable_thinking: bool | None = None,
) -> ChatOpenAI:
    extra_body = {}
    if enable_thinking is not None:
        extra_body["enable_thinking"] = enable_thinking

    return ChatOpenAI(
        model=os.environ.get("REVIEW_VISION_MODEL"),
        api_key=_get_required_env("REVIEW_VISION_API_KEY"),
        base_url=os.environ.get("REVIEW_VISION_BASE_URL"),
        temperature=float(os.environ["REVIEW_VISION_TEMPERATURE"]),
        extra_body=extra_body,
        callbacks=[AGENT_TRACE_CALLBACK],
        tags=["model:vision"],
        **_transport_options(
            timeout_seconds=int(
                os.getenv(
                    "REVIEW_VISION_TIMEOUT",
                    str(_VISION_MODEL_TIMEOUT_SECONDS),
                )
            ),
            max_retries=int(
                os.getenv("REVIEW_VISION_MAX_RETRIES", str(_MODEL_MAX_RETRIES))
            ),
        ),
    )


def build_text_model(
    *,
    parallel_tool_calls: bool | None = None,
    enable_thinking: bool | None = None,
) -> ChatOpenAI:
    model_kwargs = {}
    if parallel_tool_calls is not None:
        model_kwargs["parallel_tool_calls"] = parallel_tool_calls
    extra_body = {}
    if enable_thinking is not None:
        extra_body["enable_thinking"] = enable_thinking

    return ChatOpenAI(
        model=os.environ.get("REVIEW_TEXT_MODEL"),
        api_key=_get_required_env("REVIEW_TEXT_API_KEY"),
        base_url=os.environ.get("REVIEW_TEXT_BASE_URL"),
        temperature=float(os.environ["REVIEW_TEXT_TEMPERATURE"]),
        model_kwargs=model_kwargs,
        extra_body=extra_body,
        callbacks=[AGENT_TRACE_CALLBACK],
        tags=["model:text"],
        **_transport_options(),
    )
