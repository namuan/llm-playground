import json
import logging
import os
from typing import Dict, List, Union

import requests


# https://github.com/jmorganca/ollama/blob/main/docs/api.md
class OllamaProvider:
    def get_available_models(self) -> List[str]:
        logging.debug("get_available_models")
        try:
            response = requests.get(
                f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
                timeout=5,
            )
            logging.debug("API Response: %s", response.text)
            response.raise_for_status()
            response_json = json.loads(response.text)
            return [m["name"] for m in response_json["models"]]
        except requests.RequestException as err:
            raise err

    #
    def generate_completion(
        self, model: str, prompt: str, **kwargs
    ) -> Dict[str, Union[str, List[str]]]:
        params = {"model": model, "prompt": prompt}
        params.update(kwargs)
        logging.debug(f"generate_completion: {json.dumps(params)}")
        try:
            response = requests.post(
                f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(params),
            )
            logging.debug("API Response: %s", response.text)
            response.raise_for_status()
        except requests.RequestException as err:
            return {"error": f"API call error: {str(err)}"}

        try:
            output = "".join(
                [
                    json.loads(line)["response"]
                    for line in response.text.split("\n")
                    if line.strip() and "response" in json.loads(line)
                ]
            )

            return {"output": output.strip()}
        except json.JSONDecodeError as err:
            return {
                "error": f"API response error: {str(err)}: {json.dumps(response.text)}"
            }


if __name__ == "__main__":
    PROMPT = """What is the capital of France?"""

    llm_provider = OllamaProvider()
    llm_response = llm_provider.generate_completion("mistral:latest", PROMPT)
    print(llm_response)
    print(llm_provider.get_available_models())
