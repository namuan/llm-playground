#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pyautogen"
# ]
# ///
import random

import autogen

# config_list = [{"model": "gpt-4", "api_base": "http://0.0.0.0:8000", "api_key": "NULL"}]
config_list = [{"model": "gpt-4"}]

seed = random.randint(0, 1000000)
print(f"seed: {seed}")

assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config={
        "seed": seed,
        "config_list": config_list,
        "temperature": 0,
    },
)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "target", "use_docker": False},
)

user_proxy.initiate_chat(
    assistant,
    message="""
    Create a Plot to show the percent gain of MSFT, AAPL, TSLA, NVDA, NFLX, and GOOG stocks for each day in the year-to-date period
    And save it in a file called "stocks-gains.png"
    """,
)
