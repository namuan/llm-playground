#!/usr/bin/env python3
"""
Playing with OpenAI tools and function to call database
https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
"""

import json
import os
import sqlite3
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

import openai
import requests
from dotenv import load_dotenv
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from termcolor import colored

from logger import setup_logging

load_dotenv()

GPT_MODEL = "gpt-3.5-turbo-1106"
openai.api_key = os.getenv("OPENAI_API_KEY")

DB_FILE = Path.cwd() / "target" / "chinook.db"


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    return parser.parse_args()


def get_table_names(conn):
    """Return a list of table names"""
    table_names = []
    tables = conn.execute("SELECT name from sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names"""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database"""
    table_dicts = []
    for table_name in get_table_names(conn):
        column_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": column_names})
    return table_dicts


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(
                colored(
                    f"system: {message['content']}\n", role_to_color[message["role"]]
                )
            )
        elif message["role"] == "user":
            print(
                colored(f"user: {message['content']}\n", role_to_color[message["role"]])
            )
        elif message["role"] == "assistant" and message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['function_call']}\n",
                    role_to_color[message["role"]],
                )
            )
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['content']}\n", role_to_color[message["role"]]
                )
            )
        elif message["role"] == "tool":
            print(
                colored(
                    f"function ({message['name']}): {message['content']}\n",
                    role_to_color[message["role"]],
                )
            )


def build_tools_schema(database_schema_string):
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Use this function to answer user questions about music. Input should be a fully formed SQL query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def column_names_as_string(column_names):
    return ", ".join(column_names)


def ask_database(conn, query):
    """Function to query database with the model generated SQL query"""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"Query {query} failed with error {e}"

    return results


def execute_function_call(conn, message):
    if message["tool_calls"][0]["function"]["name"] == "ask_database":
        query = json.loads(message["tool_calls"][0]["function"]["arguments"])["query"]
        results = ask_database(conn, query)
    else:
        results = f"Error: function {message['tool_calls'][0]['function']['name']} does not exust"

    return results


def main(args):
    conn = sqlite3.connect(DB_FILE)
    database_schema_info = get_database_info(conn)
    database_schema_string = "\n".join(
        [
            f"Table: {table['table_name']}\nColumns: {column_names_as_string(table['column_names'])}"
            for table in database_schema_info
        ]
    )

    tools = build_tools_schema(database_schema_string)

    messages = []
    messages.append(
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. "
            "Ask for clarification if a user request is ambiguous."
            "Answer user questions by generating SQL query against the database",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "What is the name of the album with the most tracks?",
        }
    )
    chat_response = chat_completion_request(
        messages,
        tools=tools,
    )
    assistant_message = chat_response.json()["choices"][0]["message"]
    messages.append(assistant_message)
    print(assistant_message)
    if assistant_message.get("tool_calls"):
        results = execute_function_call(conn, assistant_message)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": assistant_message["tool_calls"][0]["id"],
                "name": assistant_message["tool_calls"][0]["function"]["name"],
                "content": results,
            }
        )
    pretty_print_conversation(messages)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
