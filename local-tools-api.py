import json

from openai import OpenAI


def get_flight_times(departure: str, arrival: str) -> str:
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

    key = f"{departure}-{arrival}".upper()
    return json.dumps(flights.get(key, {"error": "Flight not found"}))


provided_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_times",
            "description": "Get the flight times between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "The departure city (airport code)",
                    },
                    "arrival": {
                        "type": "string",
                        "description": "The arrival city (airport code)",
                    },
                },
                "required": ["departure", "arrival"],
            },
        },
    },
]

available_functions = {
    "get_flight_times": get_flight_times,
}


system_prompt = f"""
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments as follows:
{{"name": <function-name>,"arguments": <args-dict>}}

Here are the available tools:
<tools>
{provided_tools[0]}
</tools>
"""


messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": "What is the flight time from New York (NYC) to Los Angeles (LAX)?",
    },
]


def call_runtime(model: str, base_url, runtime):
    client = OpenAI(base_url=base_url, api_key=runtime)

    # First API call: Send the query and function description to the model
    response_message = (
        client.chat.completions.create(
            model=model,
            messages=messages,
            tools=provided_tools,
        )
        .choices[0]
        .message
    )
    print(f"{runtime} response -> {response_message}")

    # # Check if the model decided to use the provided function
    if not response_message.tool_calls:
        print(f"⚠️ {runtime} -> The model didn't use the function. Its response was:")
        print(response_message.content)
        return

    tool_to_use = response_message.tool_calls[0]
    function_to_call = available_functions[tool_to_use.function.name]
    function_arguments = json.loads(tool_to_use.function.arguments)
    function_response = function_to_call(
        function_arguments["departure"],
        function_arguments["arrival"],
    )
    print(f"{runtime} function call response -> {function_response}")


call_runtime("llama3.1:latest", base_url="http://localhost:11434/v1", runtime="ollama")
call_runtime(
    "meta-llama-3.1-8b-instruct",
    base_url="http://localhost:1234/v1",
    runtime="lm-studio",
)
