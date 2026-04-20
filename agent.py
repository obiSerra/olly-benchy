import json

from agent_tools import get_tools, execute_tool
from utils import call_ollama_chat, setup_logger

# Configure logger for this module
logger = setup_logger(__name__)


class OneShotAgent:
    def __init__(self, model_name, ollama_config, tool_registry=None):
        self.model_name = model_name
        self.ollama_config = ollama_config
        self.tool_registry = tool_registry if tool_registry is not None else []

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        # Generate a concise tool list with only names and descriptions

        tools = get_tools(self.tool_registry)
        tool_list = ""
        if tools:
            tool_list = "\n".join(
                [
                    f"- {tool['function']['name']}: {tool['function']['description']}"
                    for tool in tools
                ]
            )
        else:
            tool_list = "No tools available"

        system_message = f"""
You are a helpful assistant that can call tools to get information or perform actions. 
Your specialization is related to the tool list provided:
{tool_list}
When you need to use a tool, respond with a JSON object in the following format:
{{
    "tool": "tool_name",
    "arguments": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
"""
        return system_message

    def run_agent_loop(self, user_input):
        # Combine system prompt and user input
        logger.info("Starting agent loop")
        logger.info(f"User input: {user_input}")
        logger.debug(f"System prompt: {self.system_prompt}")

        done = False

        messages = [{"role": "user", "content": user_input}]
        performances = []
        tool_calls = []
        iteration = 0

        while not done:
            iteration += 1
            logger.info(f"Agent iteration {iteration}")

            response = call_ollama_chat(
                self.model_name,
                messages,
                system_prompt=self.system_prompt,
                options=self.ollama_config.get("options", {}),
            )
            if response is None:
                logger.error("LLM call failed, aborting agent loop")
                return None

            performances.append(response["performance"])

            assistant_message = response.get("raw", {}).get("message", {})
            tool_calls_in_response = assistant_message.get("tool_calls", [])
            logger.debug(f"Assistant message: {json.dumps(assistant_message, indent=2)}")
            logger.info(f"Tool calls in response: {len(tool_calls_in_response)}")

            if tool_calls_in_response:
                logger.info(f"Processing {len(tool_calls_in_response)} tool call(s)")
                # Execute each tool call
                for idx, tool_call in enumerate(tool_calls_in_response, 1):
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    logger.info(f"Tool call {idx}/{len(tool_calls_in_response)}: {tool_name}")
                    tool_calls.append({"tool": tool_name, "arguments": tool_args})
                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_args, self.tool_registry)

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_result,
                        }
                    )
                    logger.debug(f"Tool result added to message history")

                # Continue loop to let model process tool results
                logger.info("Continuing agent loop to process tool results")
            else:
                logger.info("No tool calls, agent loop complete")
                done = True  # No more tool calls, exit loop

        logger.info(f"Agent loop completed after {iteration} iteration(s)")
        logger.info(f"Total tool calls: {len(tool_calls)}")
        logger.debug(f"Final response: {response.get('response', '')}")

        return {
            "response": response.get("response", ""),
            "performances": performances,
            "tool_calls": tool_calls,
        }
