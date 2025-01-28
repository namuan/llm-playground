#!/usr/bin/env python3
"""
Ollama with functions using outline framework
"""

import json
from typing import Callable
from typing import Dict
from typing import List

import outlines
import requests
import rich
from pydantic import BaseModel
from pydantic import Field


def generate_full_completion(model: str, prompt: str, **kwargs) -> dict[str, str]:
    params = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(params),
            timeout=60,
        )
        # print(f"🤖 Request: {json.dumps(params)} -> Response: {response.text}")
        response.raise_for_status()
        return json.loads(response.text)
    except requests.RequestException as err:
        return {"error": f"API call error: {str(err)}"}


class Article:
    pass


class Weather:
    pass


class Directions:
    pass


def calculate_mortgage_payment(
    loan_amount: int, interest_rate: float, loan_term: int
) -> float:
    """Get the monthly mortgage payment given an interest rate percentage."""


def get_article_details(
    title: str,
    authors: list[str],
    short_summary: str,
    date_published: str,
    tags: list[str],
) -> Article:
    '''Get article details from unstructured article text.
    date_published: formatted as "MM/DD/YYYY"'''


def get_weather(city: str) -> Weather:
    """Get the current weather given a city."""


def get_directions(start: str, destination: str) -> Directions:
    """Get directions from Google Directions API.
    start: start address as a string including zipcode (if any)
    destination: end address as a string including zipcode (if any)"""


class ToolMetadata(BaseModel):
    """
    Tool metadata is a description of a tool that can be used to generate a response.
    """

    tool: str = Field(description="Name of the tool")
    tool_input: Dict[str, str] = Field(description="Tool input arguments")


@outlines.prompt
def tools_prompt_generator(tools: List[Callable], response_model) -> str:
    """
    You have access to the following tools:

    {% for tool in tools %}
    Tool Name: {{ tool | name }}
    Tool Description: {{ tool | description }}
    Tool Arguments: {{ tool | signature }}
    {% endfor %}

    Guidelines which MUST be followed:
    Rule 1: Always select one or more of the above tools based on the user query
    Rule 2: It is possible that there is no tool that match the user request. In that case, just respond with the following JSON without adding any Notes or Explanation otherwise my kittens will be killed:
    []

    Rule 3: If a tool is found, you must respond in the JSON format matching the following schema:
    {{ response_model | schema }}

    Rule 4: If there are multiple tools required, make sure a list of tools are returned in a JSON array.
    Rule 5: Do not add any additional Notes or Explanations

    User Query:
    """


def main():
    functions_prompt = tools_prompt_generator(
        [get_weather, get_directions, get_article_details, calculate_mortgage_payment],
        ToolMetadata,
    )
    GPT_MODEL = "mistral"

    prompts = [
        "What's the weather in London, UK?",
        "Determine the monthly mortgage payment for a loan amount of $200,000, an interest rate of 4%, and a loan term of 30 years.",
        "What's the current exchange rate for GBP to EUR?",
        "I'm planning a trip to Killington, Vermont (05751) from Hoboken, NJ (07030). Can you get me weather for both locations and directions?",
    ]

    for prompt in prompts:
        print(f"❓{prompt}")
        question = functions_prompt + prompt
        response = generate_full_completion(GPT_MODEL, question)
        try:
            tidy_response = (
                response.get("response", response)
                .strip()
                .replace("\n", "")
                .replace("\\", "")
            )
            json.loads(tidy_response)
            # print_json(tidy_response)
            rich.print(
                f"[bold]Total duration: {int(response.get('total_duration')) / 1e9} seconds [/bold]"
            )
        except Exception:
            print(f"❌ Unable to decode JSON. {tidy_response}")


if __name__ == "__main__":
    main()
