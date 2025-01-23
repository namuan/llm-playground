#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "ollama",
# ]
# ///
"""
A script to demonstrate function chaining with Ollama LLM

Usage:
./ollama-function-chaining.py -h

./ollama-function-chaining.py -v # To log INFO messages
./ollama-function-chaining.py -vv # To log DEBUG messages
"""
import json
import logging
import random
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import ollama


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


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

    def __init__(self):
        self.functions = {
            "get_weather_forecast": get_weather_forecast,
            "get_random_city": get_random_city,
        }
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

    def call_function(self, function):
        """Call the function from the given input."""

        def check_if_input_is_output(input: dict) -> dict:
            for key, value in input.items():
                if value in self.outputs:
                    input[key] = self.outputs[value]
            return input

        function_name = function["name"]
        logging.info(f"Calling function: {function_name}")

        function_input = function["params"] if "params" in function else None
        function_input = (
            check_if_input_is_output(function_input) if function_input else None
        )

        output = (
            self.functions[function_name](**function_input)
            if function_input
            else self.functions[function_name]()
        )
        self.outputs[function["output"]] = output
        return output


def main(args):
    function_caller = FunctionCaller()
    functions_metadata = function_caller.create_functions_metadata()

    prompt_beginning = """
    You are an AI assistant that can help the user with a variety of tasks. You have access to the following functions:
    """

    system_prompt_end = """
    When the user asks you a question, if you need to use functions, provide ONLY the function calls, and NOTHING ELSE, in the format:
    <function_calls>
    [
        { "name": "function_name_1", "params": { "param_1": "value_1", "param_2": "value_2" }, "output": "The output variable name, to be possibly used as input for another function},
        { "name": "function_name_2", "params": { "param_3": "value_3", "param_4": "output_1"}, "output": "The output variable name, to be possibly used as input for another function"},
        ...
    ]
    """
    system_prompt = (
        prompt_beginning
        + f"<tools> {json.dumps(functions_metadata, indent=4)} </tools>"
        + system_prompt_end
    )

    user_query = "Can you get me the weather forecast for a random city?"
    logging.info(f"Processing user query: {user_query}")

    model_name = "llama3.1:latest"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_query},
    ]

    response = ollama.chat(model=model_name, messages=messages)
    logging.debug(f"LLM response: {response}")

    function_calls = response["message"]["content"]
    if function_calls.startswith("<function_calls>"):
        function_calls = function_calls.split("<function_calls>")[1]

    try:
        function_calls_json: list[dict[str, str]] = json.loads(function_calls)
    except json.JSONDecodeError:
        function_calls_json = []
        logging.error("Model response not in desired JSON format")
    finally:
        logging.info(f"Function calls: {function_calls_json}")

    output = ""
    for function in function_calls_json:
        output = f"Agent Response: {function_caller.call_function(function)}"

    logging.info(f"Final output: {output}")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
