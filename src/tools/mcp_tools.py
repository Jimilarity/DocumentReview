import os
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_SERVER_ENV = {
    "law_retrieval_semantic": (
        "LAW_RETRIEVAL_SEMANTIC_URL",
        "LAW_RETRIEVAL_SEMANTIC_KEY",
    ),
    "law_retrieval_keyword": (
        "LAW_RETRIEVAL_KEYWORD_URL",
        "LAW_RETRIEVAL_KEYWORD_KEY",
    ),
}


def build_mcp_config() -> dict[str, dict[str, Any]]:
    """根据环境变量构建当前可用的 MCP 服务配置。"""

    config: dict[str, dict[str, Any]] = {}
    for server_name, (url_variable, key_variable) in MCP_SERVER_ENV.items():
        url = os.getenv(url_variable)
        if not url:
            continue

        server_config: dict[str, Any] = {
            "transport": "http",
            "url": url,
        }
        api_key = os.getenv(key_variable)
        if api_key:
            server_config["headers"] = {
                "Authorization": f"Bearer {api_key}"
            }
        config[server_name] = server_config

    return config


async def load_mcp_tools() -> list[Any]:
    """加载已配置的 MCP 工具；没有配置服务时返回空列表。"""

    config = build_mcp_config()
    if not config:
        return []

    client = MultiServerMCPClient(config)
    return list(await client.get_tools())
