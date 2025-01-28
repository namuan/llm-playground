#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "ollama",
#   "pydantic",
# ]
# ///
"""
A simple example to demonstrate function chaining with Ollama LLM

Usage:
./ollama-function-chaining.py -h

./ollama-function-chaining.py -v # To log INFO messages
./ollama-function-chaining.py -vv # To log DEBUG messages
"""

import json
import logging
import random
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from typing import Dict
from typing import List

import ollama
from pydantic import BaseModel

from logger import setup_logging


class Function(BaseModel):
    name: str
    params: Dict[str, str]
    output: str


class Pipeline(BaseModel):
    functions: List[Function]


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
        help="Model name to use for chat",
    )
    return parser.parse_args()


def get_weather_forecast(location: str) -> dict[str, str]:
    """Retrieves the weather forecast for a given location"""
    logging.debug(f"Getting weather forecast for {location}")
    # Mock values for test
    return {
        "location": location,
        "forecast": "sunny",
        "temperature": "25°C",
    }


def get_random_city() -> str:
    """Retrieves a random city from a list of cities"""
    cities = [
        "Groningen",
        "Enschede",
        "Amsterdam",
        "Istanbul",
        "Baghdad",
        "Rio de Janeiro",
        "Tokyo",
        "Kampala",
    ]
    city = random.choice(cities)
    logging.debug(f"Selected random city: {city}")
    return city


class FunctionCaller:
    """A class to call functions from tools.py."""

    def __init__(self, functions):
        self.functions = {func.__name__: func for func in functions}
        self.outputs = {}

    def create_functions_metadata(self) -> list[dict]:
        """Creates the functions metadata for the prompt."""

        def format_type(p_type: str) -> str:
            if p_type.startswith("<class"):
                p_type = p_type.split("'")[1]
            return p_type

        functions_metadata = []
        for name, function in self.functions.items():
            descriptions = function.__doc__.split("\n")
            logging.debug(f"Creating metadata for function: {name}")
            functions_metadata.append(
                {
                    "name": name,
                    "description": descriptions[0],
                    "parameters": (
                        {
                            "properties": [
                                {
                                    "name": param_name,
                                    "type": format_type(str(param_type)),
                                }
                                for param_name, param_type in function.__annotations__.items()
                                if param_name != "return"
                            ],
                            "required": [
                                param_name
                                for param_name in function.__annotations__
                                if param_name != "return"
                            ],
                        }
                        if function.__annotations__
                        else {}
                    ),
                    "returns": [
                        {
                            "name": name + "_output",
                            "type": {
                                param_name: format_type(str(param_type))
                                for param_name, param_type in function.__annotations__.items()
                                if param_name == "return"
                            }["return"],
                        }
                    ],
                }
            )

        return functions_metadata

    def call_function(self, function: Function):
        """Call the function from the given input."""

        def check_if_input_is_output(input: dict) -> dict:
            for key, value in input.items():
                if value in self.outputs:
                    input[key] = self.outputs[value]
            return input

        function_name = function.name
        logging.info(f"Calling function: {function_name}")

        function_input = (
            check_if_input_is_output(function.params) if function.params else None
        )

        output = (
            self.functions[function_name](**function_input)
            if function_input
            else self.functions[function_name]()
        )
        self.outputs[function.output] = output
        return output


def main(args):
    logging.info(f"Using model {args.model}")
    function_caller = FunctionCaller([get_weather_forecast, get_random_city])
    functions_metadata = function_caller.create_functions_metadata()

    system_prompt = f"""
    You are an AI assistant that can help the user with a variety of tasks. You have access to the following functions:
    <tools> {json.dumps(functions_metadata, indent=4)} </tools>

    When the user asks you a question, if you need to use functions, provide ONLY the function calls, and NOTHING ELSE, in the format:
    [
        {{ "name": "function_name_1", "params": {{ "param_1": "value_1", "param_2": "value_2" }}, "output": "The output variable name, to be possibly used as input for another function"}},
        {{ "name": "function_name_2", "params": {{ "param_3": "value_3", "param_4": "output_1"}}, "output": "The output variable name, to be possibly used as input for another function"}},
        ...
    ]
    """

    user_query = "Can you get me the weather forecast for a random city?"
    logging.info(f"Processing user query: {user_query}")

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_query},
    ]

    response = ollama.chat(
        model=args.model,
        messages=messages,
        format=Pipeline.model_json_schema(),
    )
    logging.debug(f"LLM response: {response}")

    function_pipeline = Pipeline.model_validate_json(response.message.content)
    print(function_pipeline)

    output = ""
    for function in function_pipeline.functions:
        output = f"Agent Response: {function_caller.call_function(function)}"

    logging.info(f"Final output: {output}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
