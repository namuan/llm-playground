#!/usr/bin/env python3
"""
A script to evaluate Ollama's function calling capabilities for flight time queries.

Usage:
./ollama-tools-eval.py -h

./ollama-tools-eval.py -v # To log INFO messages
./ollama-tools-eval.py -vv # To log DEBUG messages
./ollama-tools-eval.py -m llama2:13b # To specify a different model
"""

import inspect
import json
import logging
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from typing import Any
from typing import Callable
from typing import Dict

import ollama

from logger import setup_logging


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
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="llama3.2:latest",
        help="Specify the full Ollama model name to use (default: llama3.2:latest)",
    )
    return parser.parse_args()


def generate_tool_definition(func: Callable) -> Dict[str, Any]:
    """
    Generate a tool definition in JSON format from a function signature.

    :param func: The function to generate the tool definition for.
    :return: A dictionary representing the tool definition.
    """
    sig = inspect.signature(func)

    # Get function name and docstring
    name = func.__name__
    description = inspect.getdoc(func) or "No description available"

    # Generate parameters
    parameters = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        param_type = "string"  # Default to string if type hint is not available
        if param.annotation != inspect.Parameter.empty:
            if param.annotation is str:
                param_type = "string"
            elif param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"
            # Add more type mappings as needed

        parameters["properties"][param_name] = {
            "type": param_type,
            "description": f"The {param_name} parameter",
        }

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    tool_definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }

    return tool_definition


def get_flight_times(departure_code: str, arrival_code: str) -> str:
    logging.info(f"get_flight_times({departure_code=}, {arrival_code=})")
    flights = {
        "NYC-LAX": {
            "departure": "08:00 AM",
            "arrival": "11:30 AM",
            "duration": "5h 30m",
        },
        "LAX-NYC": {
            "departure": "02:00 PM",
            "arrival": "10:30 PM",
            "duration": "5h 30m",
        },
        "LHR-JFK": {
            "departure": "10:00 AM",
            "arrival": "01:00 PM",
            "duration": "8h 00m",
        },
        "JFK-LHR": {
            "departure": "09:00 PM",
            "arrival": "09:00 AM",
            "duration": "7h 00m",
        },
        "CDG-DXB": {
            "departure": "11:00 AM",
            "arrival": "08:00 PM",
            "duration": "6h 00m",
        },
        "DXB-CDG": {
            "departure": "03:00 AM",
            "arrival": "07:30 AM",
            "duration": "7h 30m",
        },
    }

    key = f"{departure_code}-{arrival_code}".upper()
    return json.dumps(flights.get(key, {"error": "Flight not found"}))


def run_ollama_chat(model: str):
    client = ollama.Client()
    messages = [
        {
            "role": "user",
            "content": "What is the flight time from New York (NYC) to Los Angeles (LAX)?",
        }
    ]

    logging.info(f"Sending initial query to {model}")
    response = client.chat(
        model=model,
        messages=messages,
        tools=[
            generate_tool_definition(get_flight_times),
        ],
    )

    messages.append(response["message"])

    if not response["message"].get("tool_calls"):
        logging.warning("The model didn't use the function. Its response was:")
        logging.warning(response["message"]["content"])
        return

    logging.info("❓ Processing function calls made by the model")
    if response["message"].get("tool_calls"):
        available_functions = {
            "get_flight_times": get_flight_times,
        }
        for tool in response["message"]["tool_calls"]:
            function_to_call = available_functions[tool["function"]["name"]]
            function_response = function_to_call(
                tool["function"]["arguments"]["departure_code"],
                tool["function"]["arguments"]["arrival_code"],
            )
            logging.info(f"✅ Function response: {function_response}")
            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                }
            )

    # logging.info("Getting final response from the model")
    # final_response = client.chat(model=model, messages=messages)
    # logging.info("Final response:")
    # print(final_response["message"]["content"])


def main(args):
    logging.debug(f"Verbosity level: {args.verbose}")
    logging.info(f"Using model: {args.model}")
    run_ollama_chat(args.model)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
