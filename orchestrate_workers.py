#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "ollama",
#   "pydantic",
# ]
# ///
import json
from typing import Literal, List
import concurrent.futures
import ollama
from pydantic import BaseModel, Field, ValidationError

ORCHESTRATOR_PROMPT = """
Analyze this task and break it down into 2-3 distinct approaches:

Task: {task}

Provide an Analysis:

Explain your understanding of the task and which variations would be valuable.
Focus on how each approach serves different aspects of the task.

Along with the analysis, provide 2-3 approaches to tackle the task, each with a brief description:

Formal style: Write technically and precisely, focusing on detailed specifications
Conversational style: Write in a friendly and engaging way that connects with the reader
Hybrid style: Tell a story that includes technical details, combining emotional elements with specifications

Return only JSON output.
"""

WORKER_PROMPT = """
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return only your response:
[Your content here, maintaining the specified style and fully addressing requirements.]
"""

task = """Write a product description for a new eco-friendly water bottle.
The target_audience is environmentally conscious millennials and key product features are: plastic-free, insulated, lifetime warranty
"""


class Task(BaseModel):
    type: Literal["formal", "conversational", "hybrid"]
    description: str


class TaskList(BaseModel):
    analysis: str
    tasks: List[Task] = Field(..., default_factory=list)


def json_llm(user_prompt: str, schema, system_prompt: str = None):
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        response = ollama.chat(
            model="llama3.2:latest",
            messages=messages,
            format=schema.model_json_schema(),
        )

        return schema.model_validate_json(response.message.content)

    except ValidationError as e:
        error_message = f"Failed to parse JSON: {e}"
        print(error_message)


def run_llm(user_prompt: str, model: str, system_prompt: str = None):
    print(f"Running {model} with {user_prompt}")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    response = ollama.chat(
        model=model, messages=messages, options={"temperature": 0.7, "max_tokens": 4000}
    )

    return response.message.content


def orchestrator_workflow(task: str, orchestrator_prompt: str, worker_prompt: str):
    """Use a orchestrator model to break down a task into sub-tasks and then use worker models to generate and return responses."""

    # Use orchestrator model to break the task up into sub-tasks
    orchestrator_response = json_llm(
        orchestrator_prompt.format(task=task), schema=TaskList
    )

    # Parse orchestrator response
    analysis = orchestrator_response.analysis
    tasks: TaskList = orchestrator_response.tasks

    print("\n=== ORCHESTRATOR OUTPUT ===")
    print(f"\nANALYSIS:\n{analysis}")
    print(f"\nTASKS:\n{json.dumps([t.dict() for t in tasks], indent=2)}")
    # print(f"\nTASKS:\n{json.dumps(tasks.model_dump(), indent=2)}")

    worker_model = ["qwen2.5:latest"] * len(tasks)

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_llm,
                user_prompt=worker_prompt.format(
                    original_task=task,
                    task_type=task_info.type,
                    task_description=task_info.description,
                ),
                model=model,
            )
            for task_info, model in zip(tasks, worker_model)
        ]
        worker_resp = [f.result() for f in concurrent.futures.as_completed(futures)]

    return tasks, worker_resp


def main():
    task = """Write a product description for a new eco-friendly water bottle.
    The target_audience is environmentally conscious millennials and key product features are: plastic-free, insulated, lifetime warranty
    """

    tasks, worker_resp = orchestrator_workflow(
        task, orchestrator_prompt=ORCHESTRATOR_PROMPT, worker_prompt=WORKER_PROMPT
    )

    for task_info, response in zip(tasks, worker_resp):
        print(f"\n=== WORKER RESULT ({task_info.type}) ===\n{response}\n")


if __name__ == "__main__":
    main()
