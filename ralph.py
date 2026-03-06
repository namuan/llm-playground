#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = []
# ///
"""
Ralph Wiggum Loop for AI agents - Python implementation

Implementation of the Ralph Wiggum technique - continuous self-referential
AI loops for iterative development. Based on ghuntley.com/ralph/

Usage:
  ralph.py "<prompt>" [options]
  ralph.py --prompt-file <path> [options]
  ralph.py -h
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

VERSION = "1.2.0"

STATE_DIR = Path.cwd() / ".ralph"
STATE_PATH = STATE_DIR / "ralph-loop.state.json"
CONTEXT_PATH = STATE_DIR / "ralph-context.md"
HISTORY_PATH = STATE_DIR / "ralph-history.json"
TASKS_PATH = STATE_DIR / "ralph-tasks.md"


@dataclass
class Task:
    text: str
    status: str
    subtasks: list["Task"] = field(default_factory=list)
    original_line: str = ""


@dataclass
class IterationHistory:
    iteration: int
    started_at: str
    ended_at: str
    duration_ms: int
    model: str
    tools_used: dict[str, int]
    files_modified: list[str]
    exit_code: int
    completion_detected: bool
    errors: list[str]


@dataclass
class RalphHistory:
    iterations: list[IterationHistory] = field(default_factory=list)
    total_duration_ms: int = 0
    repeated_errors: dict[str, int] = field(default_factory=dict)
    no_progress_iterations: int = 0
    short_iterations: int = 0


@dataclass
class RalphState:
    active: bool
    iteration: int
    min_iterations: int
    max_iterations: int
    completion_promise: str
    abort_promise: str
    tasks_mode: bool
    task_promise: str
    prompt: str
    prompt_template: str
    started_at: str
    model: str


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B\[[0-9;]*m", "", text)


def escape_regex(s: str) -> str:
    return re.escape(s)


def format_duration(ms: int) -> str:
    total_seconds = max(0, ms // 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_duration_long(ms: int) -> str:
    total_seconds = max(0, ms // 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def parse_tool_output(line: str) -> Optional[str]:
    stripped = strip_ansi(line)
    match = re.match(r"^\|\s{2}([A-Za-z0-9_-]+)", stripped)
    return match.group(1) if match else None


def format_tool_summary(tool_counts: dict[str, int], max_items: int = 6) -> str:
    if not tool_counts:
        return ""
    entries = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    shown = entries[:max_items]
    remaining = len(entries) - len(shown)
    parts = [f"{name} {count}" for name, count in shown]
    if remaining > 0:
        parts.append(f"+{remaining} more")
    return " • ".join(parts)


def collect_tool_summary(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        tool = parse_tool_output(line)
        if tool:
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def load_state() -> Optional[RalphState]:
    if not STATE_PATH.exists():
        return None
    try:
        data = json.loads(STATE_PATH.read_text())
        return RalphState(
            active=data.get("active", False),
            iteration=data.get("iteration", 1),
            min_iterations=data.get("minIterations", 1),
            max_iterations=data.get("maxIterations", 0),
            completion_promise=data.get("completionPromise", "COMPLETE"),
            abort_promise=data.get("abortPromise", ""),
            tasks_mode=data.get("tasksMode", False),
            task_promise=data.get("taskPromise", "READY_FOR_NEXT_TASK"),
            prompt=data.get("prompt", ""),
            prompt_template=data.get("promptTemplate", ""),
            started_at=data.get("startedAt", ""),
            model=data.get("model", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def save_state(state: RalphState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "active": state.active,
        "iteration": state.iteration,
        "minIterations": state.min_iterations,
        "maxIterations": state.max_iterations,
        "completionPromise": state.completion_promise,
        "abortPromise": state.abort_promise,
        "tasksMode": state.tasks_mode,
        "taskPromise": state.task_promise,
        "prompt": state.prompt,
        "promptTemplate": state.prompt_template,
        "startedAt": state.started_at,
        "model": state.model,
    }
    STATE_PATH.write_text(json.dumps(data, indent=2))


def clear_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def load_history() -> RalphHistory:
    if not HISTORY_PATH.exists():
        return RalphHistory()
    try:
        data = json.loads(HISTORY_PATH.read_text())
        iterations = [
            IterationHistory(
                iteration=i.get("iteration", 0),
                started_at=i.get("startedAt", ""),
                ended_at=i.get("endedAt", ""),
                duration_ms=i.get("durationMs", 0),
                model=i.get("model", ""),
                tools_used=i.get("toolsUsed", {}),
                files_modified=i.get("filesModified", []),
                exit_code=i.get("exitCode", 0),
                completion_detected=i.get("completionDetected", False),
                errors=i.get("errors", []),
            )
            for i in data.get("iterations", [])
        ]
        return RalphHistory(
            iterations=iterations,
            total_duration_ms=data.get("totalDurationMs", 0),
            repeated_errors=data.get("struggleIndicators", {}).get(
                "repeatedErrors", {}
            ),
            no_progress_iterations=data.get("struggleIndicators", {}).get(
                "noProgressIterations", 0
            ),
            short_iterations=data.get("struggleIndicators", {}).get(
                "shortIterations", 0
            ),
        )
    except (json.JSONDecodeError, KeyError):
        return RalphHistory()


def save_history(history: RalphHistory) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "iterations": [
            {
                "iteration": i.iteration,
                "startedAt": i.started_at,
                "endedAt": i.ended_at,
                "durationMs": i.duration_ms,
                "model": i.model,
                "toolsUsed": i.tools_used,
                "filesModified": i.files_modified,
                "exitCode": i.exit_code,
                "completionDetected": i.completion_detected,
                "errors": i.errors,
            }
            for i in history.iterations
        ],
        "totalDurationMs": history.total_duration_ms,
        "struggleIndicators": {
            "repeatedErrors": history.repeated_errors,
            "noProgressIterations": history.no_progress_iterations,
            "shortIterations": history.short_iterations,
        },
    }
    HISTORY_PATH.write_text(json.dumps(data, indent=2))


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def load_context() -> Optional[str]:
    if not CONTEXT_PATH.exists():
        return None
    content = CONTEXT_PATH.read_text().strip()
    return content if content else None


def clear_context() -> None:
    if CONTEXT_PATH.exists():
        CONTEXT_PATH.unlink()


def parse_tasks(content: str) -> list[Task]:
    tasks: list[Task] = []
    current_task: Optional[Task] = None

    for line in content.splitlines():
        top_match = re.match(r"^- \[([ x/])\]\s*(.+)", line)
        if top_match:
            if current_task:
                tasks.append(current_task)
            status = (
                "complete"
                if top_match.group(1) == "x"
                else "in-progress"
                if top_match.group(1) == "/"
                else "todo"
            )
            current_task = Task(
                text=top_match.group(2), status=status, original_line=line
            )
            continue

        sub_match = re.match(r"^\s+- \[([ x/])\]\s*(.+)", line)
        if sub_match and current_task:
            status = (
                "complete"
                if sub_match.group(1) == "x"
                else "in-progress"
                if sub_match.group(1) == "/"
                else "todo"
            )
            current_task.subtasks.append(
                Task(text=sub_match.group(2), status=status, original_line=line)
            )

    if current_task:
        tasks.append(current_task)

    return tasks


def find_current_task(tasks: list[Task]) -> Optional[Task]:
    for task in tasks:
        if task.status == "in-progress":
            return task
    return None


def find_next_task(tasks: list[Task]) -> Optional[Task]:
    for task in tasks:
        if task.status == "todo":
            return task
    return None


def all_tasks_complete(tasks: list[Task]) -> bool:
    return len(tasks) > 0 and all(t.status == "complete" for t in tasks)


def get_tasks_mode_section(state: RalphState) -> str:
    if not TASKS_PATH.exists():
        return """
## TASKS MODE: Enabled (no tasks file found)

Create .ralph/ralph-tasks.md with your task list, or use `ralph.py --add-task "description"` to add tasks.
"""

    try:
        tasks_content = TASKS_PATH.read_text()
        tasks = parse_tasks(tasks_content)
        current_task = find_current_task(tasks)
        next_task = find_next_task(tasks)

        task_instructions = ""
        if current_task:
            task_instructions = f"""
 CURRENT TASK: "{current_task.text}"
    Focus on completing this specific task.
    When done: Mark as [x] in .ralph/ralph-tasks.md and output <promise>{state.task_promise}</promise>"""
        elif next_task:
            task_instructions = f"""
 NEXT TASK: "{next_task.text}"
    Mark as [/] in .ralph/ralph-tasks.md before starting.
    When done: Mark as [x] and output <promise>{state.task_promise}</promise>"""
        elif all_tasks_complete(tasks):
            task_instructions = f"""
 ALL TASKS COMPLETE!
    Output <promise>{state.completion_promise}</promise> to finish."""
        else:
            task_instructions = """
 No tasks found. Add tasks to .ralph/ralph-tasks.md or use `ralph.py --add-task`"""

        return f"""
## TASKS MODE: Working through task list

Current tasks from .ralph/ralph-tasks.md:
```markdown
{tasks_content.strip()}
```
{task_instructions}

### Task Workflow
1. Find any task marked [/] (in progress). If none, pick the first [ ] task.
2. Mark the task as [/] in ralph-tasks.md before starting.
3. Complete the task.
4. Mark as [x] when verified complete.
5. Output <promise>{state.task_promise}</promise> to move to the next task.
6. Only output <promise>{state.completion_promise}</promise> when ALL tasks are [x].

---
"""
    except Exception:
        return """
## TASKS MODE: Error reading tasks file

Unable to read .ralph/ralph-tasks.md
"""


def build_prompt(state: RalphState) -> str:
    if state.prompt_template and Path(state.prompt_template).exists():
        template = Path(state.prompt_template).read_text()
        context = load_context() or ""
        tasks_content = ""
        if state.tasks_mode and TASKS_PATH.exists():
            tasks_content = TASKS_PATH.read_text()
        template = template.replace("{{iteration}}", str(state.iteration))
        template = template.replace(
            "{{max_iterations}}",
            str(state.max_iterations) if state.max_iterations > 0 else "unlimited",
        )
        template = template.replace("{{min_iterations}}", str(state.min_iterations))
        template = template.replace("{{prompt}}", state.prompt)
        template = template.replace("{{completion_promise}}", state.completion_promise)
        template = template.replace("{{abort_promise}}", state.abort_promise)
        template = template.replace("{{task_promise}}", state.task_promise)
        template = template.replace("{{context}}", context)
        template = template.replace("{{tasks}}", tasks_content)
        return template

    context = load_context()
    context_section = (
        f"""
## Additional Context (added by user mid-loop)

{context}

---
"""
        if context
        else ""
    )

    if state.tasks_mode:
        tasks_section = get_tasks_mode_section(state)
        return f"""
# Ralph Wiggum Loop - Iteration {state.iteration}

You are in an iterative development loop working through a task list.
{context_section}{tasks_section}
## Your Main Goal

{state.prompt}

## Critical Rules

- Work on ONE task at a time from .ralph/ralph-tasks.md
- ONLY output <promise>{state.task_promise}</promise> when the current task is complete and marked in ralph-tasks.md
- ONLY output <promise>{state.completion_promise}</promise> when ALL tasks are truly done
- Output promise tags DIRECTLY - do not quote them, explain them, or say you "will" output them
- Do NOT lie or output false promises to exit the loop
- If stuck, try a different approach
- Check your work before claiming completion

## Current Iteration: {state.iteration}{f" / {state.max_iterations}" if state.max_iterations > 0 else " (unlimited)"} (min: {state.min_iterations})

Tasks Mode: ENABLED - Work on one task at a time from ralph-tasks.md

Now, work on the current task. Good luck!
""".strip()

    return f"""
# Ralph Wiggum Loop - Iteration {state.iteration}

You are in an iterative development loop. Work on the task below until you can genuinely complete it.
{context_section}
## Your Task

{state.prompt}

## Instructions

1. Read the current state of files to understand what's been done
2. Track your progress and plan remaining work
3. Make progress on the task
4. Run tests/verification if applicable
5. When the task is GENUINELY COMPLETE, output:
   <promise>{state.completion_promise}</promise>

## Critical Rules

- ONLY output <promise>{state.completion_promise}</promise> when the task is truly done
- Output the promise tag DIRECTLY - do not quote it, explain it, or say you "will" output it
- Do NOT lie or output false promises to exit the loop
- If stuck, try a different approach
- Check your work before claiming completion
- The loop will continue until you succeed

## Current Iteration: {state.iteration}{f" / {state.max_iterations}" if state.max_iterations > 0 else " (unlimited)"} (min: {state.min_iterations})

Now, work on the task. Good luck!
""".strip()


def check_completion(output: str, promise: str) -> bool:
    escaped = escape_regex(promise)
    pattern = re.compile(rf"<promise>\s*{escaped}\s*</promise>", re.IGNORECASE)
    matches = list(pattern.finditer(output))
    if not matches:
        return False

    for match in matches:
        context_before = output[max(0, match.start() - 100) : match.start()].lower()
        negation_patterns = [
            r"\bnot\s+(yet\s+)?(say|output|write|respond|print)",
            r"\bdon'?t\s+(say|output|write|respond|print)",
            r"\bwon'?t\s+(say|output|write|respond|print)",
            r"\bwill\s+not\s+(say|output|write|respond|print)",
            r"\bshould\s+not\s+(say|output|write|respond|print)",
            r"\bwouldn'?t\s+(say|output|write|respond|print)",
            r"\bavoid\s+(saying|outputting|writing)",
            r"\bwithout\s+(saying|outputting|writing)",
            r"\bbefore\s+(saying|outputting|I\s+say)",
            r"\buntil\s+(I\s+)?(say|output|can\s+say)",
        ]
        has_negation = any(re.search(p, context_before) for p in negation_patterns)
        if has_negation:
            continue
        quotes_before = len(re.findall(r"""["'`]""", context_before))
        if quotes_before % 2 == 1:
            continue
        return True

    return False


def load_plugins_from_config(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    try:
        raw = config_path.read_text()
        without_block = re.sub(r"/\*[\s\S]*?\*/", "", raw)
        cleaned = re.sub(r"^\s*//.*$", "", without_block, flags=re.MULTILINE)
        parsed = json.loads(cleaned)
        plugins = parsed.get("plugin", [])
        return [p for p in plugins if isinstance(p, str)]
    except (json.JSONDecodeError, TypeError):
        return []
    try:
        raw = config_path.read_text()
        without_block = re.sub(r"/\*[\s\S]*?\*/", "", raw)
        without_line = re.sub(r"^\s*//.*$", "", without_line, flags=re.MULTILINE)
        parsed = json.loads(without_line)
        plugins = parsed.get("plugin", [])
        return [p for p in plugins if isinstance(p, str)]
    except (json.JSONDecodeError, TypeError):
        return []


def ensure_ralph_config(
    filter_plugins: bool = False, allow_all_permissions: bool = True
) -> str:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = STATE_DIR / "ralph-opencode.config.json"
    xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    user_config_path = Path(xdg_config) / "opencode" / "opencode.json"
    project_config_path = Path.cwd() / ".ralph" / "opencode.json"
    legacy_project_config_path = Path.cwd() / ".opencode" / "opencode.json"

    config: dict = {"$schema": "https://opencode.ai/config.json"}

    if filter_plugins:
        plugins = [
            *load_plugins_from_config(user_config_path),
            *load_plugins_from_config(project_config_path),
            *load_plugins_from_config(legacy_project_config_path),
        ]
        config["plugin"] = list(
            {p for p in plugins if re.search(r"auth", p, re.IGNORECASE)}
        )

    if allow_all_permissions:
        config["permission"] = {
            "read": "allow",
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "bash": "allow",
            "task": "allow",
            "webfetch": "allow",
            "websearch": "allow",
            "codesearch": "allow",
            "todowrite": "allow",
            "todoread": "allow",
            "question": "allow",
            "lsp": "allow",
            "external_directory": "allow",
        }

    config_path.write_text(json.dumps(config, indent=2))
    return str(config_path)


def validate_agent() -> None:
    cmd = os.environ.get("RALPH_OPENCODE_BINARY", "opencode")
    if not shutil.which(cmd):
        print(f"Error: OpenCode CLI ('{cmd}') not found.", file=sys.stderr)
        sys.exit(1)


def capture_file_snapshot() -> dict[str, str]:
    files: dict[str, str] = {}
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        tracked_result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
        )
        all_files: set[str] = set()
        for line in status_result.stdout.splitlines():
            if line.strip():
                all_files.add(line[3:].strip())
        for line in tracked_result.stdout.splitlines():
            if line.strip():
                all_files.add(line.strip())

        for file in all_files:
            try:
                result = subprocess.run(
                    ["git", "hash-object", file],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    files[file] = result.stdout.strip()
            except Exception:
                pass
    except Exception:
        pass
    return files


def get_modified_files_since(
    before: dict[str, str], after: dict[str, str]
) -> list[str]:
    changed: list[str] = []
    for file, hash_val in after.items():
        if before.get(file) != hash_val:
            changed.append(file)
    for file in before:
        if file not in after:
            changed.append(file)
    return changed


def extract_errors(output: str) -> list[str]:
    errors: list[str] = []
    for line in output.splitlines():
        lower = line.lower()
        if (
            "error:" in lower
            or "failed:" in lower
            or "exception:" in lower
            or "typeerror" in lower
            or "syntaxerror" in lower
            or "referenceerror" in lower
            or ("test" in lower and "fail" in lower)
        ):
            cleaned = line.strip()[:200]
            if cleaned and cleaned not in errors:
                errors.append(cleaned)
    return errors[:10]


def print_status(show_tasks: bool = False) -> None:
    state = load_state()
    history = load_history()
    context = load_context()
    show_tasks = show_tasks or (state.tasks_mode if state else False)

    print("""
+======================================================================+
|                    Ralph Wiggum Status                               |
+======================================================================+
""")

    if state and state.active:
        elapsed = int(
            (datetime.now() - datetime.fromisoformat(state.started_at)).total_seconds()
            * 1000
        )
        elapsed_str = format_duration_long(elapsed)
        print(f" ACTIVE LOOP")
        print(
            f"   Iteration:    {state.iteration}{f' / {state.max_iterations}' if state.max_iterations > 0 else ' (unlimited)'}"
        )
        print(f"   Started:      {state.started_at}")
        print(f"   Elapsed:      {elapsed_str}")
        print(f"   Promise:      {state.completion_promise}")
        print(f"   Agent:        OpenCode")
        if state.model:
            print(f"   Model:        {state.model}")
        if state.tasks_mode:
            print(f"   Tasks Mode:   ENABLED")
            print(f"   Task Promise: {state.task_promise}")
        prompt_preview = state.prompt[:60] + ("..." if len(state.prompt) > 60 else "")
        print(f"   Prompt:       {prompt_preview}")
    else:
        print("  No active loop")

    if context:
        print(f"\n PENDING CONTEXT (will be injected next iteration):")
        print(f"   {context.replace(chr(10), chr(10) + '   ')}")

    if show_tasks:
        if TASKS_PATH.exists():
            try:
                tasks_content = TASKS_PATH.read_text()
                tasks = parse_tasks(tasks_content)
                if tasks:
                    print(f"\n CURRENT TASKS:")
                    for i, task in enumerate(tasks):
                        status_icon = (
                            "+"
                            if task.status == "complete"
                            else "~"
                            if task.status == "in-progress"
                            else "-"
                        )
                        print(f"   {i + 1}. {status_icon} {task.text}")
                        for subtask in task.subtasks:
                            sub_icon = (
                                "+"
                                if subtask.status == "complete"
                                else "~"
                                if subtask.status == "in-progress"
                                else "-"
                            )
                            print(f"      {sub_icon} {subtask.text}")
                    complete = sum(1 for t in tasks if t.status == "complete")
                    in_progress = sum(1 for t in tasks if t.status == "in-progress")
                    print(
                        f"\n   Progress: {complete}/{len(tasks)} complete, {in_progress} in progress"
                    )
                else:
                    print(f"\n CURRENT TASKS: (no tasks found)")
            except Exception:
                print(f"\n CURRENT TASKS: (error reading tasks)")
        else:
            print(f"\n CURRENT TASKS: (no tasks file found)")

    if history.iterations:
        print(f"\n HISTORY ({len(history.iterations)} iterations)")
        print(f"   Total time:   {format_duration_long(history.total_duration_ms)}")
        recent = history.iterations[-5:]
        print(f"\n   Recent iterations:")
        for iter_rec in recent:
            tools = format_tool_summary(iter_rec.tools_used, max_items=3)
            model_label = iter_rec.model or "unknown"
            print(
                f"   #{iter_rec.iteration}  {format_duration_long(iter_rec.duration_ms)}  {model_label}  {tools or 'no tools'}"
            )

        has_repeated_errors = any(c >= 2 for c in history.repeated_errors.values())
        if (
            history.no_progress_iterations >= 3
            or history.short_iterations >= 3
            or has_repeated_errors
        ):
            print(f"\n  STRUGGLE INDICATORS:")
            if history.no_progress_iterations >= 3:
                print(
                    f"   - No file changes in {history.no_progress_iterations} iterations"
                )
            if history.short_iterations >= 3:
                print(f"   - {history.short_iterations} very short iterations (< 30s)")
            top_errors = sorted(
                [(e, c) for e, c in history.repeated_errors.items() if c >= 2],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            for error, count in top_errors:
                print(f'   - Same error {count}x: "{error[:50]}..."')
            print(f"\n   Tip: Use 'ralph.py --add-context \"hint\"' to guide the agent")

    print()
    sys.exit(0)


def add_context(context_text: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    new_entry = f"\n## Context added at {timestamp}\n{context_text}\n"

    if CONTEXT_PATH.exists():
        existing = CONTEXT_PATH.read_text()
        CONTEXT_PATH.write_text(existing + new_entry)
    else:
        CONTEXT_PATH.write_text(f"# Ralph Loop Context\n{new_entry}")

    print(f" Context added for next iteration")
    print(f"   File: {CONTEXT_PATH}")

    state = load_state()
    if state and state.active:
        print(f"   Will be picked up in iteration {state.iteration + 1}")
    else:
        print(f"   Will be used when loop starts")
    sys.exit(0)


def list_tasks() -> None:
    if not TASKS_PATH.exists():
        print("No tasks file found. Use --add-task to create your first task.")
        sys.exit(0)

    try:
        tasks_content = TASKS_PATH.read_text()
        tasks = parse_tasks(tasks_content)
        if not tasks:
            print("No tasks found.")
            return
        print("Current tasks:")
        for i, task in enumerate(tasks):
            status_icon = (
                "+"
                if task.status == "complete"
                else "~"
                if task.status == "in-progress"
                else "-"
            )
            print(f"{i + 1}. {status_icon} {task.text}")
            for subtask in task.subtasks:
                sub_icon = (
                    "+"
                    if subtask.status == "complete"
                    else "~"
                    if subtask.status == "in-progress"
                    else "-"
                )
                print(f"   {sub_icon} {subtask.text}")
    except Exception as e:
        print(f"Error reading tasks file: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def add_task(description: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if TASKS_PATH.exists():
        content = TASKS_PATH.read_text().rstrip() + "\n"
    else:
        content = "# Ralph Tasks\n\n"
    content += f"- [ ] {description}\n"
    TASKS_PATH.write_text(content)
    print(f' Task added: "{description}"')
    sys.exit(0)


def remove_task(index: int) -> None:
    if not TASKS_PATH.exists():
        print("Error: No tasks file found", file=sys.stderr)
        sys.exit(1)

    try:
        content = TASKS_PATH.read_text()
        tasks = parse_tasks(content)

        if index < 1 or index > len(tasks):
            print(
                f"Error: Task index {index} is out of range (1-{len(tasks)})",
                file=sys.stderr,
            )
            sys.exit(1)

        lines = content.splitlines()
        new_lines: list[str] = []
        in_removed_task = False
        current_task_line = 0

        for line in lines:
            if re.match(r"^- \[", line):
                current_task_line += 1
                if current_task_line == index:
                    in_removed_task = True
                    continue
                else:
                    in_removed_task = False

            if in_removed_task and re.match(r"^\s+", line) and line.strip():
                continue

            new_lines.append(line)

        TASKS_PATH.write_text("\n".join(new_lines))
        print(f" Removed task {index} and its subtasks")
    except Exception as e:
        print(f"Error removing task: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt", nargs="*", help="Task description for the AI to work on"
    )
    parser.add_argument(
        "--agent", default="opencode", help="AI agent to use (default: opencode)"
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=1,
        help="Minimum iterations before completion allowed",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Maximum iterations before stopping (0 = unlimited)",
    )
    parser.add_argument(
        "--completion-promise",
        default="COMPLETE",
        help="Phrase that signals completion",
    )
    parser.add_argument(
        "--abort-promise", default="", help="Phrase that signals early abort"
    )
    parser.add_argument(
        "--tasks",
        "-t",
        action="store_true",
        help="Enable Tasks Mode for structured task tracking",
    )
    parser.add_argument(
        "--task-promise",
        default="READY_FOR_NEXT_TASK",
        help="Phrase that signals task completion",
    )
    parser.add_argument("--model", default="", help="Model to use")
    parser.add_argument(
        "--prompt-file", "--file", "-f", help="Read prompt content from a file"
    )
    parser.add_argument("--prompt-template", help="Use custom prompt template")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Buffer agent output and print at the end",
    )
    parser.add_argument(
        "--verbose-tools", action="store_true", help="Print every tool line"
    )
    parser.add_argument(
        "--no-plugins", action="store_true", help="Disable non-auth OpenCode plugins"
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Don't auto-commit after each iteration",
    )
    parser.add_argument(
        "--allow-all",
        action="store_true",
        default=True,
        help="Auto-approve all tool permissions",
    )
    parser.add_argument(
        "--no-allow-all",
        action="store_true",
        help="Require interactive permission prompts",
    )
    parser.add_argument("--version", "-v", action="store_true", help="Show version")
    parser.add_argument(
        "--status", action="store_true", help="Show current Ralph loop status"
    )
    parser.add_argument(
        "--add-context", metavar="TEXT", help="Add context for the next iteration"
    )
    parser.add_argument(
        "--clear-context", action="store_true", help="Clear any pending context"
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="Display the current task list"
    )
    parser.add_argument("--add-task", metavar="DESC", help="Add a new task to the list")
    parser.add_argument(
        "--remove-task", type=int, metavar="N", help="Remove task at index N"
    )
    parser.add_argument(
        "--",
        dest="extra_flags",
        nargs="*",
        help="Pass remaining arguments to the agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.version:
        print(f"ralph.py {VERSION}")
        sys.exit(0)

    if args.status:
        print_status(args.tasks)

    if args.add_context:
        add_context(args.add_context)

    if args.clear_context:
        clear_context()
        print(
            " Context cleared"
            if CONTEXT_PATH.exists()
            else " No pending context to clear"
        )
        sys.exit(0)

    if args.list_tasks:
        list_tasks()

    if args.add_task:
        add_task(args.add_task)

    if args.remove_task:
        remove_task(args.remove_task)

    prompt = " ".join(args.prompt) if args.prompt else ""
    prompt_template = args.prompt_template or ""

    if args.prompt_file:
        prompt_file = Path(args.prompt_file)
        if not prompt_file.exists():
            print(f"Error: Prompt file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
        prompt = prompt_file.read_text()
    elif len(args.prompt) == 1 and Path(args.prompt[0]).exists():
        prompt = Path(args.prompt[0]).read_text()
        prompt_template = prompt_template or ""

    if not prompt:
        existing = load_state()
        if existing and existing.active:
            prompt = existing.prompt
        else:
            print("Error: No prompt provided", file=sys.stderr)
            print('Usage: ralph.py "Your task description" [options]', file=sys.stderr)
            sys.exit(1)

    if args.max_iterations > 0 and args.min_iterations > args.max_iterations:
        print(
            f"Error: --min-iterations ({args.min_iterations}) cannot be greater than --max-iterations ({args.max_iterations})",
            file=sys.stderr,
        )
        sys.exit(1)

    allow_all = not args.no_allow_all
    validate_agent()

    existing_state = load_state()
    resuming = existing_state.active if existing_state else False

    if resuming and existing_state:
        state = existing_state
    else:
        state = RalphState(
            active=True,
            iteration=1,
            min_iterations=args.min_iterations,
            max_iterations=args.max_iterations,
            completion_promise=args.completion_promise,
            abort_promise=args.abort_promise,
            tasks_mode=args.tasks,
            task_promise=args.task_promise,
            prompt=prompt,
            prompt_template=prompt_template,
            started_at=datetime.now().isoformat(),
            model=args.model,
        )
        save_state(state)

    if args.tasks and not TASKS_PATH.exists():
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        TASKS_PATH.write_text(
            '# Ralph Tasks\n\nAdd your tasks below using: `ralph.py --add-task "description"`\n'
        )
        print(f" Created tasks file: {TASKS_PATH}")

    history = load_history() if resuming else RalphHistory()
    if not resuming:
        save_history(history)

    prompt_preview = prompt.replace("\n", " ")[:80] + (
        "..." if len(prompt) > 80 else ""
    )
    print(f"Task: {prompt_preview}")
    print(f"Completion promise: {args.completion_promise}")
    if args.tasks:
        print(f"Tasks mode: ENABLED")
        print(f"Task promise: {args.task_promise}")
    print(f"Min iterations: {args.min_iterations}")
    print(
        f"Max iterations: {args.max_iterations if args.max_iterations > 0 else 'unlimited'}"
    )
    print(f"Agent: OpenCode")
    if args.model:
        print(f"Model: {args.model}")
    if args.no_plugins:
        print("OpenCode plugins: non-auth plugins disabled")
    if allow_all:
        print("Permissions: auto-approve all tools")
    print()
    print("Starting loop... (Ctrl+C to stop)")
    print("=" * 68)

    current_proc: Optional[subprocess.Popen] = None
    stopping = False

    def handle_sigint(signum, frame):
        nonlocal stopping
        if stopping:
            print("\nForce stopping...")
            sys.exit(1)
        stopping = True
        print("\nGracefully stopping Ralph loop...")
        if current_proc:
            try:
                current_proc.terminate()
            except Exception:
                pass
        clear_state()
        print("Loop cancelled.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    extra_flags = args.extra_flags or []

    while True:
        if state.max_iterations > 0 and state.iteration > state.max_iterations:
            print(
                f"\n+======================================================================+"
            )
            print(f"|  Max iterations ({state.max_iterations}) reached. Loop stopped.")
            print(f"|  Total time: {format_duration_long(history.total_duration_ms)}")
            print(
                f"+======================================================================+"
            )
            clear_state()
            break

        iter_info = f" / {state.max_iterations}" if state.max_iterations > 0 else ""
        min_info = (
            f" (min: {state.min_iterations})"
            if state.min_iterations > 1 and state.iteration < state.min_iterations
            else ""
        )
        print(f"\n Iteration {state.iteration}{iter_info}{min_info}")
        print("-" * 68)

        context_at_start = load_context()
        snapshot_before = capture_file_snapshot()
        full_prompt = build_prompt(state)
        iteration_start = time.time()

        try:
            cmd_args = ["run"]
            if state.model:
                cmd_args.extend(["-m", state.model])
            if extra_flags:
                cmd_args.extend(extra_flags)
            cmd_args.append(full_prompt)

            env = os.environ.copy()
            if args.no_plugins or allow_all:
                env["OPENCODE_CONFIG"] = ensure_ralph_config(
                    filter_plugins=args.no_plugins,
                    allow_all_permissions=allow_all,
                )

            opencode_bin = os.environ.get("RALPH_OPENCODE_BINARY", "opencode")
            current_proc = subprocess.Popen(
                [opencode_bin] + cmd_args,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout_text, stderr_text = current_proc.communicate()
            exit_code = current_proc.returncode
            current_proc = None

            combined_output = stdout_text + "\n" + stderr_text
            tool_counts = collect_tool_summary(combined_output)

            if not args.no_stream:
                if stderr_text:
                    print(stderr_text, file=sys.stderr)
                print(stdout_text)

            completion_detected = check_completion(
                combined_output, state.completion_promise
            )
            abort_detected = (
                check_completion(combined_output, state.abort_promise)
                if state.abort_promise
                else False
            )
            task_completion_detected = (
                check_completion(combined_output, state.task_promise)
                if state.tasks_mode
                else False
            )

            iteration_duration = int((time.time() - iteration_start) * 1000)

            print("\nIteration Summary")
            print("-" * 68)
            print(f"Iteration: {state.iteration}")
            print(f"Elapsed:   {format_duration(iteration_duration)}")
            print(f"Tools:     {format_tool_summary(tool_counts) or 'none'}")
            print(f"Exit code: {exit_code}")
            print(
                f"Completion promise: {'detected' if completion_detected else 'not detected'}"
            )

            snapshot_after = capture_file_snapshot()
            files_modified = get_modified_files_since(snapshot_before, snapshot_after)
            errors = extract_errors(combined_output)

            iter_record = IterationHistory(
                iteration=state.iteration,
                started_at=datetime.fromtimestamp(iteration_start).isoformat(),
                ended_at=datetime.now().isoformat(),
                duration_ms=iteration_duration,
                model=state.model,
                tools_used=tool_counts,
                files_modified=files_modified,
                exit_code=exit_code,
                completion_detected=completion_detected,
                errors=errors,
            )
            history.iterations.append(iter_record)
            history.total_duration_ms += iteration_duration

            if not files_modified:
                history.no_progress_iterations += 1
            else:
                history.no_progress_iterations = 0

            if iteration_duration < 30000:
                history.short_iterations += 1
            else:
                history.short_iterations = 0

            if not errors:
                history.repeated_errors = {}
            else:
                for error in errors:
                    key = error[:100]
                    history.repeated_errors[key] = (
                        history.repeated_errors.get(key, 0) + 1
                    )

            save_history(history)

            if state.iteration > 2 and (
                history.no_progress_iterations >= 3 or history.short_iterations >= 3
            ):
                print(f"\n Potential struggle detected:")
                if history.no_progress_iterations >= 3:
                    print(
                        f"   - No file changes in {history.no_progress_iterations} iterations"
                    )
                if history.short_iterations >= 3:
                    print(f"   - {history.short_iterations} very short iterations")
                print(
                    f"   Tip: Use 'ralph.py --add-context \"hint\"' in another terminal to guide the agent"
                )

            if exit_code != 0:
                print(
                    f"\n  OpenCode exited with code {exit_code}. Continuing to next iteration."
                )

            if abort_detected:
                print(
                    f"\n+======================================================================+"
                )
                print(
                    f"|   Abort signal detected: <promise>{state.abort_promise}</promise>"
                )
                print(f"|  Loop aborted after {state.iteration} iteration(s)")
                print(
                    f"|  Total time: {format_duration_long(history.total_duration_ms)}"
                )
                print(
                    f"+======================================================================+"
                )
                clear_state()
                clear_history()
                clear_context()
                sys.exit(1)

            if task_completion_detected and not completion_detected:
                print(
                    f"\n Task completion detected: <promise>{state.task_promise}</promise>"
                )
                print(f"   Moving to next task in iteration {state.iteration + 1}...")

            if completion_detected:
                if state.iteration < state.min_iterations:
                    print(
                        f"\n Completion promise detected, but minimum iterations ({state.min_iterations}) not yet reached."
                    )
                    print(f"   Continuing to iteration {state.iteration + 1}...")
                else:
                    print(
                        f"\n+======================================================================+"
                    )
                    print(
                        f"|   Completion promise detected: <promise>{state.completion_promise}</promise>"
                    )
                    print(f"|  Task completed in {state.iteration} iteration(s)")
                    print(
                        f"|  Total time: {format_duration_long(history.total_duration_ms)}"
                    )
                    print(
                        f"+======================================================================+"
                    )
                    clear_state()
                    clear_history()
                    clear_context()
                    break

            if context_at_start:
                print(f" Context was consumed this iteration")
                clear_context()

            if not args.no_commit:
                try:
                    status_result = subprocess.run(
                        ["git", "status", "--porcelain"],
                        capture_output=True,
                        text=True,
                    )
                    if status_result.stdout.strip():
                        subprocess.run(["git", "add", "-A"], capture_output=True)
                        subprocess.run(
                            [
                                "git",
                                "commit",
                                "-m",
                                f"Ralph iteration {state.iteration}: work in progress",
                            ],
                            capture_output=True,
                        )
                        print(f" Auto-committed changes")
                except Exception:
                    pass

            state.iteration += 1
            save_state(state)
            time.sleep(1)

        except Exception as e:
            if current_proc:
                try:
                    current_proc.terminate()
                except Exception:
                    pass
                current_proc = None
            print(f"\n Error in iteration {state.iteration}: {e}")
            print("Continuing to next iteration...")

            iteration_duration = int((time.time() - iteration_start) * 1000)
            error_record = IterationHistory(
                iteration=state.iteration,
                started_at=datetime.fromtimestamp(iteration_start).isoformat(),
                ended_at=datetime.now().isoformat(),
                duration_ms=iteration_duration,
                model=state.model,
                tools_used={},
                files_modified=[],
                exit_code=-1,
                completion_detected=False,
                errors=[str(e)[:200]],
            )
            history.iterations.append(error_record)
            history.total_duration_ms += iteration_duration
            save_history(history)

            state.iteration += 1
            save_state(state)
            time.sleep(2)


if __name__ == "__main__":
    main()
