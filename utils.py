import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import requests


def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with file rotation.
    
    Args:
        name: Name for the logger (typically __name__ of the module)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Extract module name from the logger name for the filename
    module_name = name.split('.')[-1] if '.' in name else name
    
    # Create rotating file handler with date in filename
    log_filename = log_dir / f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


# Configure logger for this module
logger = setup_logger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


def call_ollama_base(
    model: str,
    api_endpoint: str,
    payload: dict,
) -> Optional[dict]:
    logger.info(f"Calling Ollama API: {api_endpoint}")
    logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(api_endpoint, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Raw response: {json.dumps(result, indent=2)}")

        # Extract performance metrics
        total_duration = result.get("total_duration", 0) / 1e9
        load_duration = result.get("load_duration", 0) / 1e9
        prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9
        eval_duration = result.get("eval_duration", 0) / 1e9
        eval_count = result.get("eval_count", 0)

        # Calculate tokens per second
        tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0

        # Build performance dictionary
        performance = {
            "total_duration_s": total_duration,
            "load_duration_s": load_duration,
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "prompt_eval_duration_s": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration_s": eval_duration,
            "tokens_per_second": tokens_per_second,
            "model": result.get("model", model),
            "created_at": result.get("created_at", ""),
        }

        logger.info(f"LLM Response - Tokens: {eval_count}, Speed: {tokens_per_second:.2f} tok/s, Duration: {total_duration:.2f}s")
        logger.debug(f"Response content: {result.get('response', '')[:200]}...")

        return {
            "response": result.get("response", ""),
            "performance": performance,
            "raw": result,
        }

    except requests.Timeout:
        logger.error(f"Request timed out after 300 seconds")
        print(f"Error: Request timed out after 300 seconds")
        return None
    except requests.RequestException as e:
        logger.error(f"Error calling Ollama API: {e}")
        print(f"Error calling Ollama API: {e}")
        return None


def call_ollama_generate(
    model: str,
    prompt: str,
    options: Optional[dict] = None,
    ollama_base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Optional[dict]:
    logger.info(f"Calling Ollama generate - Model: {model}")
    logger.debug(f"Prompt: {prompt}")
    if system_prompt:
        logger.debug(f"System prompt: {system_prompt}")

    base_url = ollama_base_url or OLLAMA_BASE_URL

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Disable streaming to get full response at once
    }

    if options is not None:
        payload["options"] = options

    if system_prompt is not None:
        payload["system"] = system_prompt

    return call_ollama_base(model, url, payload)


def call_ollama_chat(
    model: str,
    messages: list,
    options: Optional[dict] = None,
    ollama_base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Optional[dict]:
    logger.info(f"Calling Ollama chat - Model: {model}, Messages: {len(messages)}")
    logger.debug(f"Chat messages: {json.dumps(messages, indent=2)}")
    if system_prompt:
        logger.debug(f"System prompt: {system_prompt}")

    base_url = ollama_base_url or OLLAMA_BASE_URL

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "stream": False,  # Disable streaming to get full response at once
        "messages": messages,
    }

    if options is not None:
        payload["options"] = options

    if system_prompt is not None:
        payload["system"] = system_prompt

    return call_ollama_base(model, url, payload)
