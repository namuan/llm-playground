from litellm import completion

# ollama_response = completion(
#     model="ollama/llama2-uncensored",
#     messages=[{"content": "Write a helpful review comment?", "role": "user"}],
#     api_base="http://localhost:11434",
#     stream=True,
# )
#
# for chunk in ollama_response:
#     print(chunk["choices"][0]["delta"]["content"], end="")

import openai

openai.api_base = "http://0.0.0.0:8000"

print(
    openai.ChatCompletion.create(
        model="test", messages=[{"role": "user", "content": "Write a code review comment?"}]
    )
)
