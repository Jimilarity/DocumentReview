from typing import Annotated, Any, Dict, List, Optional, TypedDict




def dedupe_tools(tools: List[Any]) -> List[Any]:
    seen_names = set()
    unique_tools = []

    for current_tool in tools:
        name = getattr(current_tool, "name", None)

        if name is not None and name in seen_names:
            print(f"skip duplicate tool: {name}")
            continue

        if name is not None:
            seen_names.add(name)

        unique_tools.append(current_tool)

    return unique_tools