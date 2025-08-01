#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
#   "jsonschema",
# ]
# ///
# Description: A script to interact with Ollama via LiteLLM using file manipulation tools.
# Example: ./ollama-agent.py [-v | -vv | -vvv] [--model MODEL]
import argparse
import json
import logging
import os
import sys
from logging import Logger
from typing import Tuple, Optional, Callable, Any

import litellm


def configure_logging(verbose_count: int) -> Logger:
    """Configure logging based on verbose count."""
    levels = {
        0: logging.ERROR,   # No -v
        1: logging.WARNING, # -v
        2: logging.INFO,    # -vv
        3: logging.DEBUG    # -vvv
    }
    level = levels.get(verbose_count, logging.DEBUG)  # Default to DEBUG for >3
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def generate_schema(python_type: Any) -> dict:
    """Generate JSON schema for a given Python type with descriptions."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Generating schema for {python_type.__name__}")
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    for key, value in python_type.__annotations__.items():
        docstring = python_type.__dict__.get(key, f"The {key} parameter")
        if isinstance(docstring, str) and not docstring.startswith("The "):
            description = docstring
        else:
            description = f"The {key} parameter"
        schema["properties"][key] = {
            "type": "string",
            "description": description
        }
        if key != "path" or python_type.__name__ != "ListFilesInput":
            schema["required"].append(key)
    logger.debug(f"Generated schema: {json.dumps(schema)}")
    return schema

class ReadFileInput:
    path: str  # The relative path of a file in the working directory.

class ListFilesInput:
    path: str  # Optional relative path to list files from. Defaults to current directory if not provided.

class EditFileInput:
    path: str    # The path to the file
    old_str: str # Text to search for - must match exactly and must only have one match exactly
    new_str: str # Text to replace old_str with

class ToolDefinition:
    def __init__(self, name: str, description: str, input_schema: dict, function: Callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.function = function

def read_file(input_data: dict) -> tuple[str, Optional[str]]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Executing read_file with input: {input_data}")
    try:
        with open(input_data["path"], "r") as f:
            content = f.read()
        logger.debug(f"Read file content: {content[:100]}...")  # Log first 100 chars
        return content, None
    except Exception as e:
        logger.error(f"Error in read_file: {str(e)}")
        return "", str(e)

def list_files(input_data: dict) -> tuple[str, Optional[str]]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Executing list_files with input: {input_data}")
    try:
        dir_path = input_data.get("path", ".")
        files = []
        for root, dirs, filenames in os.walk(dir_path):
            for name in filenames + dirs:
                rel_path = os.path.relpath(os.path.join(root, name), dir_path)
                if os.path.isdir(os.path.join(root, name)):
                    files.append(rel_path + "/")
                else:
                    files.append(rel_path)
        result = json.dumps(files)
        logger.debug(f"List files result: {result}")
        return result, None
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        return "", str(e)

def edit_file(input_data: dict) -> tuple[str, Optional[str]]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Executing edit_file with input: {input_data}")
    try:
        if input_data["path"] == "" or input_data["old_str"] == input_data["new_str"]:
            logger.error("Invalid input parameters for edit_file")
            return "", "Invalid input parameters"

        try:
            with open(input_data["path"], "r") as f:
                content = f.read()
        except FileNotFoundError:
            if input_data["old_str"] == "":
                os.makedirs(os.path.dirname(input_data["path"]) or ".", exist_ok=True)
                with open(input_data["path"], "w") as f:
                    f.write(input_data["new_str"])
                logger.debug(f"Created file {input_data['path']}")
                return f"Successfully created file {input_data['path']}", None
            logger.error("File not found in edit_file")
            return "", "File not found"

        new_content = content.replace(input_data["old_str"], input_data["new_str"])
        if new_content == content and input_data["old_str"] != "":
            logger.error("old_str not found in file")
            return "", "old_str not found in file"

        with open(input_data["path"], "w") as f:
            f.write(new_content)
        logger.debug("File edit successful")
        return "OK", None
    except Exception as e:
        logger.error(f"Error in edit_file: {str(e)}")
        return "", str(e)

def get_tool_definitions() -> list:
    """Return list of tool definitions."""
    read_file_schema = generate_schema(ReadFileInput)
    list_files_schema = generate_schema(ListFilesInput)
    edit_file_schema = generate_schema(EditFileInput)

    return [
        ToolDefinition(
            name="read_file",
            description="Read the contents of a given relative file path. Use this to see what's inside a file. Do not use with directory names.",
            input_schema=read_file_schema,
            function=read_file
        ),
        ToolDefinition(
            name="list_files",
            description="List files and directories at a given path. If no path is provided, lists files in the current directory.",
            input_schema=list_files_schema,
            function=list_files
        ),
        ToolDefinition(
            name="edit_file",
            description="Make edits to a text file. Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different. If the file doesn't exist, it will be created if old_str is empty.",
            input_schema=edit_file_schema,
            function=edit_file
        )
    ]

def main():
    parser = argparse.ArgumentParser(description="Interact with Ollama via LiteLLM")
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity: -v for WARNING, -vv for INFO, -vvv for DEBUG"
    )
    parser.add_argument(
        "--model",
        default="ollama_chat/qwen3:4b",
        help="Specify the Ollama model to use (default: ollama_chat/qwen3:4b)"
    )
    args = parser.parse_args()

    logger = configure_logging(args.verbose)
    logger.debug("Starting main function")

    def get_user_message() -> Tuple[Optional[str], bool]:
        logger = logging.getLogger(__name__)
        try:
            user_input = input("\033[94mYou\033[0m: ")
            logger.debug(f"Received user input: {user_input}")
            return user_input, True
        except EOFError:
            logger.debug("Received EOF, exiting")
            return None, False

    agent_run(get_user_message, get_tool_definitions(), args.model)

def agent_run(get_user_message: Callable, tools: list, model: str):
    logger = logging.getLogger(__name__)
    conversation = []
    logger.debug("Starting agent run")
    print("Chat with Ollama (use 'ctrl-c' to quit)")
    read_user_input = True

    while True:
        if read_user_input:
            user_input, ok = get_user_message()
            if not ok:
                logger.debug("No user input, breaking loop")
                break

            conversation.append({"role": "user", "content": user_input})
            logger.debug(f"Appended user message to conversation: {user_input}")

        response = run_inference(conversation, tools, model)
        logger.debug(f"Received response: {response}")
        content = response["content"]
        # Serialize content to string if it's a list
        serialized_content = json.dumps(content) if isinstance(content, list) else content
        conversation.append({"role": "assistant", "content": serialized_content})
        logger.debug(f"Appended assistant response to conversation: {serialized_content}")

        tool_results = []
        for content_block in content if isinstance(content, list) else [content]:
            logger.debug(f"Processing content block: {content_block}")
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                print(f"\033[93mOllama\033[0m: {content_block['text']}")
            elif isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                logger.debug(f"Tool use detected: {content_block['name']} with input {content_block['input']}")
                result = execute_tool(content_block["tool_use_id"], content_block["name"], content_block["input"], tools)
                logger.debug(f"Tool execution result: {result}")
                tool_results.append(result)

        if not tool_results:
            logger.debug("No tool results, continuing to read user input")
            read_user_input = True
            continue

        read_user_input = False
        conversation.append({"role": "user", "content": json.dumps(tool_results)})
        logger.debug(f"Appended tool results to conversation: {tool_results}")

def run_inference(conversation: list, tools: list, model: str):
    logger = logging.getLogger(__name__)
    tools_spec = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        } for tool in tools
    ]
    logger.debug(f"Tools specification: {tools_spec}")
    try:
        # Convert any list-based content to string before sending to API
        modified_conversation = []
        for message in conversation:
            modified_message = message.copy()
            if isinstance(message["content"], list):
                modified_message["content"] = json.dumps(message["content"])
            modified_conversation.append(modified_message)

        response = litellm.completion(
            model=model,
            messages=modified_conversation,
            api_base="http://localhost:11434",
            tools=tools_spec
        )
        logger.debug(f"Raw response from LiteLLM: {response}")
        content_blocks = []
        choice = response.choices[0]
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                logger.debug(f"Processing tool call: {tool_call}")
                content_blocks.append({
                    "type": "tool_use",
                    "tool_use_id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)
                })
                logger.debug(f"Processed tool_use content: {content_blocks[-1]}")
        else:
            content_blocks.append({"type": "text", "text": choice.message.content or ""})
            logger.debug(f"Processed text content: {content_blocks[-1]}")
        return {"content": content_blocks}
    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}")
        return {"content": [{"type": "text", "text": f"Error: {str(e)}"}]}

def execute_tool(tool_use_id: str, name: str, input_data: dict, tools: list):
    logger = logging.getLogger(__name__)
    logger.debug(f"Attempting to execute tool: {name} with input: {input_data}")
    for tool in tools:
        if tool.name == name:
            print(f"\033[92mtool\033[0m: {name}({json.dumps(input_data)})")
            result, error = tool.function(input_data)
            logger.debug(f"Tool {name} executed with result: {result}, error: {error}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "is_error": bool(error)
            }
    logger.warning(f"Tool not found: {name}")
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": "tool not found",
        "is_error": True
    }

if __name__ == "__main__":
    main()