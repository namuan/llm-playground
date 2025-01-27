#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "ollama==0.4.7",
#   "pydantic",
# ]
# ///
from ollama import chat
from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


response = chat(
    messages=[
        {
            "role": "user",
            "content": "Tell me about Canada.",
        }
    ],
    model="llama3.1",
    format=Country.model_json_schema(),
)

country = Country.model_validate_json(response.message.content)
print(country)
