from litellm import completion

print("Using LangChain with Ollama directly")

ollama_response = completion(
    model="ollama/llama2-uncensored:latest",
    messages=[{"content": "Write a helpful review comment?", "role": "user"}],
    api_base="http://localhost:11434",
    stream=True,
)

for chunk in ollama_response:
    print(chunk["choices"][0]["delta"]["content"], end="")

print("")
print("Using OpenAI connecting to Ollama via LiteLLM Proxy")

from openai import OpenAI

client = OpenAI(base_url="http://0.0.0.0:8000")

chat_completion = client.chat.completions.create(
    model="ollama/zephyr:latest", messages=[{"role": "user", "content": "Write a code review comment?"}]
)
print(chat_completion.choices[0].message.content)
