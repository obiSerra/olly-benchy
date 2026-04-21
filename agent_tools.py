"""Tools module for the AI agent.

Contains tool definitions, registration system, and helper functions.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Optional


# Import after other imports to avoid circular dependency
from utils import setup_logger

# Configure logger for this module
logger = setup_logger(__name__)

# Global context for file operations
_artifacts_dir = None


def set_artifacts_directory(artifacts_dir: Path):
    """Set the artifacts directory for file operations.
    
    Args:
        artifacts_dir: Path to the artifacts directory for this run
    """
    global _artifacts_dir
    _artifacts_dir = artifacts_dir
    logger.info(f"Set artifacts directory to: {artifacts_dir}")


def get_artifacts_directory() -> Optional[Path]:
    """Get the current artifacts directory.
    
    Returns:
        Path to the artifacts directory, or None if not set
    """
    return _artifacts_dir


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


# File management tools

def create_file(filename: str, content: str) -> dict[str, Any]:
    """Create a new file in the artifacts directory.
    
    Args:
        filename: Name of the file to create (relative path within artifacts directory)
        content: Content to write to the file
        
    Returns:
        Dictionary with success status and message
    """
    if _artifacts_dir is None:
        logger.error("Artifacts directory not set")
        return {"success": False, "error": "Artifacts directory not configured"}
    
    try:
        # Resolve the full path
        file_path = _artifacts_dir / filename
        
        # Security check: ensure the file is within the artifacts directory
        if not file_path.resolve().is_relative_to(_artifacts_dir.resolve()):
            logger.error(f"Security violation: attempted to write outside artifacts directory: {filename}")
            return {"success": False, "error": "Cannot write outside artifacts directory"}
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        file_path.write_text(content)
        logger.info(f"Created file: {file_path}")
        
        return {
            "success": True, 
            "message": f"File created: {filename}",
            "path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error creating file {filename}: {e}")
        return {"success": False, "error": str(e)}


def read_file(filepath: str) -> dict[str, Any]:
    """Read the contents of any file.
    
    Args:
        filepath: Path to the file to read (can be absolute or relative)
        
    Returns:
        Dictionary with file contents or error message
    """
    try:
        file_path = Path(filepath)
        
        # If it's not absolute, try relative to current directory
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}
        
        # Read the file
        content = file_path.read_text()
        logger.info(f"Read file: {file_path} ({len(content)} characters)")
        
        return {
            "success": True,
            "content": content,
            "path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return {"success": False, "error": str(e)}


def delete_file(filename: str) -> dict[str, Any]:
    """Delete a file from the artifacts directory.
    
    Args:
        filename: Name of the file to delete (relative path within artifacts directory)
        
    Returns:
        Dictionary with success status and message
    """
    if _artifacts_dir is None:
        logger.error("Artifacts directory not set")
        return {"success": False, "error": "Artifacts directory not configured"}
    
    try:
        # Resolve the full path
        file_path = _artifacts_dir / filename
        
        # Security check: ensure the file is within the artifacts directory
        if not file_path.resolve().is_relative_to(_artifacts_dir.resolve()):
            logger.error(f"Security violation: attempted to delete outside artifacts directory: {filename}")
            return {"success": False, "error": "Cannot delete outside artifacts directory"}
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}
        
        # Delete the file
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")
        
        return {
            "success": True,
            "message": f"File deleted: {filename}",
            "path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return {"success": False, "error": str(e)}


# Tool registry by name
AVAILABLE_TOOLS = {
    "create_file": create_file,
    "read_file": read_file,
    "delete_file": delete_file,
}


def get_tool_by_name(tool_name: str):
    """Get a tool function by its name.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        The tool function, or None if not found
    """
    return AVAILABLE_TOOLS.get(tool_name)
