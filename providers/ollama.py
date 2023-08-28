import json
import logging
import os
from typing import Dict, List, Union

import requests


class OllamaProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def id(self) -> str:
        return f"ollama:{self.model_name}"

    def __str__(self) -> str:
        return f"[Ollama Provider {self.model_name}]"

    # https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-completion
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Union[str, List[str]]]:
        params = {"model": self.model_name, "prompt": prompt}
        params.update(kwargs)
        logging.debug(f"Calling Ollama API: {json.dumps(params)}")
        try:
            response = requests.post(
                f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(params),
                timeout=5,
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

    llm_provider = OllamaProvider(model_name="llama2:13b")
    llm_response = llm_provider.generate_completion(PROMPT)
    print(llm_response)
