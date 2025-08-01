#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests"
# ]
# ///
"""
Fetches all models from OpenRouter API and filters for free ones
$ uv run openrouter-free-models.py
Link with Aider
$ ln -s ~/workspace/llm-playground/openrouter-free-models.py ~/bin/openrouter-free-models.py
$ aider --model openrouter/$(openrouter-free-models.py | fzf | awk '{print $2}')
"""

import json
import sys

import requests


def get_free_openrouter_models(names_only=False):
    try:
        print("Fetching models from OpenRouter...")

        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError("Unexpected API response format")

        print(f"Total models found: {len(data['data'])}")

        # Filter for free models
        # Free models typically have pricing.prompt = "0" and pricing.completion = "0"
        free_models = [
            model
            for model in data["data"]
            if "pricing" in model
            and float(model["pricing"].get("prompt", "0")) == 0
            and float(model["pricing"].get("completion", "0")) == 0
        ]

        print(f"Free models found: {len(free_models)}\n")

        # Display the free models
        if free_models:
            if names_only:
                # Return only model IDs (names) one per line
                for model in free_models:
                    print(model["id"])
            else:
                for index, model in enumerate(free_models, 1):
                    name = (
                        model["name"]
                        if "name" in model and model["name"] != model["id"]
                        else model["id"]
                    )
                    context_length_k = f"{model['context_length']/1000:.0f}K"
                    print(f"- {model['id']} - {name} - Context Length: {context_length_k}")
        else:
            print("No free models found.")

        # Return the list for potential programmatic use
        return free_models

    except Exception as error:
        print(f"Error fetching models: {error}")
        return []


# Run the function if this script is executed directly
if __name__ == "__main__":
    # Check for --names-only flag
    names_only = "--names-only" in sys.argv
    free_models = get_free_openrouter_models(names_only=names_only)

    # Optional: Save to JSON file
    save_to_file = "--save" in sys.argv
    if save_to_file:
        try:
            with open("openrouter-free-models.json", "w", encoding="utf-8") as f:
                json.dump(free_models, f, indent=2)
            print("✅ Results saved to openrouter-free-models.json")
        except Exception as error:
            print(f"Failed to save file: {error}")
