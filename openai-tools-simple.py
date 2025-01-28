#!/usr/bin/env python3
"""
Playing with OpenAI tools and function
https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
"""

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from termcolor import colored

from logger import setup_logging

load_dotenv()

GPT_MODEL = "gpt-3.5-turbo-1106"

client = OpenAI()


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


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request_api(messages, tools=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools
        )
        return response.choices[0].message
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
        if isinstance(message, ChatCompletionMessage):
            message_role = message.role
            message_content = message.content
            message_name = message.function_call.name if message.function_call else None
            message_function_call = message.function_call
        else:
            message_role = message["role"]
            message_content = message["content"]
            message_name = message.get("name")
            message_function_call = message.get("function_call")

        if message_role == "system":
            print(colored(f"system: {message_content}\n", role_to_color[message_role]))
        elif message_role == "user":
            print(colored(f"user: {message_content}\n", role_to_color[message_role]))
        elif message_role == "assistant" and message_function_call:
            print(
                colored(
                    f"assistant: {message_function_call}\n",
                    role_to_color[message_role],
                )
            )
        elif message_role == "assistant" and not message_function_call:
            print(
                colored(f"assistant: {message_content}\n", role_to_color[message_role])
            )
        elif message_role == "tool":
            print(
                colored(
                    f"function ({message_name}): {message_content}\n",
                    role_to_color[message_role],
                )
            )


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["location", "format", "num_days"],
            },
        },
    },
]


def main(args):
    messages = []
    messages.append(
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. "
            "Ask for clarification if a user request is ambiguous.",
        }
    )
    # messages.append({"role": "user", "content": "What is the weather like today?"})
    # messages.append({"role": "user", "content": "I'm in London, UK"})
    messages.append(
        {
            "role": "user",
            "content": "what is the weather going to be like in London, UK and Dubai over the next 5 days",
        }
    )
    assistant_message = chat_completion_request_api(
        messages,
        tools=tools,
    )
    messages.append(assistant_message)
    print(assistant_message)
    pretty_print_conversation(messages)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
