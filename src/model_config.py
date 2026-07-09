import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(env_path)




def _get_required_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable, tried: {', '.join(names)}")


def build_vision_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("REVIEW_VISION_MODEL"),
        api_key=_get_required_env("REVIEW_VISION_API_KEY"),
        base_url=os.environ.get("REVIEW_VISION_BASE_URL"),
    )


def build_text_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("REVIEW_TEXT_MODEL"),
        api_key=_get_required_env("REVIEW_TEXT_API_KEY"),
        base_url=os.environ.get("REVIEW_TEXT_BASE_URL"),
    )
