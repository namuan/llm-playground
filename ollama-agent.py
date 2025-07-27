#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "anthropic",
#   "jsonschema",
# ]
# ///
# Description: A script to interact with Claude using file manipulation tools.
# Example: ./ollama-agent.py
import anthropic
import sys
import json
import os
from typing import Tuple, Optional, Callable, Any
from jsonschema import Draft7Validator

def generate_schema(python_type: Any) -> dict:
    """Generate JSON schema for a given Python type with descriptions."""
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

read_file_schema = generate_schema(ReadFileInput)
list_files_schema = generate_schema(ListFilesInput)
edit_file_schema = generate_schema(EditFileInput)

def read_file(input_data: dict) -> tuple[str, Optional[str]]:
    try:
        with open(input_data["path"], "r") as f:
            content = f.read()
        return content, None
    except Exception as e:
        return "", str(e)

def list_files(input_data: dict) -> tuple[str, Optional[str]]:
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
        return json.dumps(files), None
    except Exception as e:
        return "", str(e)

def edit_file(input_data: dict) -> tuple[str, Optional[str]]:
    try:
        if input_data["path"] == "" or input_data["old_str"] == input_data["new_str"]:
            return "", "Invalid input parameters"

        try:
            with open(input_data["path"], "r") as f:
                content = f.read()
        except FileNotFoundError:
            if input_data["old_str"] == "":
                os.makedirs(os.path.dirname(input_data["path"]) or ".", exist_ok=True)
                with open(input_data["path"], "w") as f:
                    f.write(input_data["new_str"])
                return f"Successfully created file {input_data['path']}", None
            return "", "File not found"

        new_content = content.replace(input_data["old_str"], input_data["new_str"])
        if new_content == content and input_data["old_str"] != "":
            return "", "old_str not found in file"

        with open(input_data["path"], "w") as f:
            f.write(new_content)
        return "OK", None
    except Exception as e:
        return "", str(e)

read_file_definition = ToolDefinition(
    name="read_file",
    description="Read the contents of a given relative file path. Use this to see what's inside a file. Do not use with directory names.",
    input_schema=read_file_schema,
    function=read_file
)

list_files_definition = ToolDefinition(
    name="list_files",
    description="List files and directories at a given path. If no path is provided, lists files in the current directory.",
    input_schema=list_files_schema,
    function=list_files
)

edit_file_definition = ToolDefinition(
    name="edit_file",
    description="Make edits to a text file. Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different. If the file doesn't exist, it will be created if old_str is empty.",
    input_schema=edit_file_schema,
    function=edit_file
)

def main():
    client = anthropic.Anthropic()

    def get_user_message() -> Tuple[Optional[str], bool]:
        try:
            user_input = input("\033[94mYou\033[0m: ")
            return user_input, True
        except EOFError:
            return None, False

    agent_run(client, get_user_message, [read_file_definition, list_files_definition, edit_file_definition])

def agent_run(client: anthropic.Anthropic, get_user_message: Callable, tools: list):
    conversation = []
    print("Chat with Claude (use 'ctrl-c' to quit)")
    read_user_input = True

    while True:
        if read_user_input:
            user_input, ok = get_user_message()
            if not ok:
                break

            conversation.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

        response = run_inference(client, conversation, tools)
        conversation.append({"role": "assistant", "content": response.content})

        tool_results = []
        for content in response.content:
            if content.type == "text":
                print(f"\033[93mClaude\033[0m: {content.text}")
            elif content.type == "tool_use":
                result = execute_tool(content.tool_use_id, content.name, content.input, tools)
                tool_results.append(result)

        if not tool_results:
            read_user_input = True
            continue

        read_user_input = False
        conversation.append({"role": "user", "content": tool_results})

def run_inference(client: anthropic.Anthropic, conversation: list, tools: list):
    tools_spec = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema
        } for tool in tools
    ]
    return client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=conversation,
        tools=tools_spec
    )

def execute_tool(tool_use_id: str, name: str, input_data: dict, tools: list):
    for tool in tools:
        if tool.name == name:
            print(f"\033[92mtool\033[0m: {name}({json.dumps(input_data)})")
            result, error = tool.function(input_data)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "is_error": bool(error)
            }
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": "tool not found",
        "is_error": True
    }

if __name__ == "__main__":
    main()