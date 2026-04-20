"""Tools module for the AI agent.

Contains tool definitions, registration system, and helper functions.
"""

import inspect
import json
from typing import Any


# Import after other imports to avoid circular dependency
from utils import setup_logger

# Configure logger for this module
logger = setup_logger(__name__)

# Registry for tool functions


def generate_tool_definition(func) -> dict[str, Any]:
    """Generate Ollama tool definition from a function's signature and docstring.

    Args:
        func: The function to generate a tool definition for.

    Returns:
        A dictionary in Ollama function calling format.
    """
    # Get function name
    name = func.__name__

    # Get docstring and parse it
    docstring = inspect.getdoc(func) or ""
    lines = docstring.split("\n")

    # Extract description (first line before Args section)
    description = lines[0] if lines else ""

    # Parse Args section
    properties = {}
    required = []
    in_args_section = False

    for line in lines:
        if line.strip().startswith("Args:"):
            in_args_section = True
            continue
        elif line.strip().startswith("Returns:"):
            in_args_section = False
            break

        if in_args_section and line.strip():
            # Parse parameter line (format: "param_name: description")
            if ":" in line:
                parts = line.strip().split(":", 1)
                param_name = parts[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""

                properties[param_name] = {
                    "type": "string",  # Default to string
                    "description": param_desc,
                }

    # Get function signature to determine required parameters
    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty and param_name in properties:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_tools(tool_registry) -> list[dict[str, Any]]:
    """Get all registered tools in Ollama function calling format.

    Returns:
        List of tool definitions.
    """
    return [generate_tool_definition(func) for func in tool_registry]


def get_tool_functions(tool_registry) -> dict[str, Any]:
    """Get mapping of tool names to functions.

    Returns:
        Dictionary mapping tool names to their implementations.
    """
    return {func.__name__: func for func in tool_registry}


def execute_tool(tool_name: str, arguments: dict[str, Any], tool_registry) -> str:
    """Execute a tool and return the result as a JSON string.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Dictionary of arguments to pass to the tool.

    Returns:
        JSON string containing the tool result.
    """
    logger.info(f"Executing tool: {tool_name}")
    logger.debug(f"Tool arguments: {json.dumps(arguments, indent=2)}")

    tool_functions = get_tool_functions(tool_registry)
    if tool_name in tool_functions:
        result = tool_functions[tool_name](**arguments)
        logger.info(f"Tool {tool_name} executed successfully")
        logger.debug(f"Tool result: {json.dumps(result, indent=2)}")
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
        logger.error(f"Unknown tool requested: {tool_name}")

    return json.dumps(result)
