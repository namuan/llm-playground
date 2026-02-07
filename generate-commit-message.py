#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests"
# ]
# ///

import sys
import json
import logging
import requests

logger = logging.getLogger("generate_commit_msg")

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

DEFAULT_PROMPT = (
    "You are a helpful assistant that generates concise, meaningful git commit messages "
    "based on the provided diff. Follow the conventional commits format (e.g., feat: add new feature, "
    "fix: resolve bug, docs: update documentation, etc.).\n\n"
    "Rules:\n"
    "- Output ONLY the commit message\n"
    "- Never include markdown formatting or code blocks\n"
    "- Never add explanatory text before or after\n"
    "- Assume output will be used directly in a git commit\n"
)

def build_messages(system_prompt: str, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def stream_sse(resp: requests.Response, verbose: int = 0):
    for raw_line in resp.iter_lines(decode_unicode=True):
        if verbose >= 2 and raw_line is not None:
            logger.debug(">> %s", raw_line)
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                if verbose >= 2:
                    logger.debug("JSON decode failed for data: %s", data)
                continue
            delta = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                yield delta

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("diff", nargs="?", default=None, help="Git diff to generate commit message for")
    args = parser.parse_args()

    if args.diff is None:
        # Read from stdin if no argument is provided
        diff_input = sys.stdin.read()
    else:
        diff_input = args.diff

    if not diff_input.strip():
        print("Error: No diff provided. Pipe your git diff or pass it as an argument.", file=sys.stderr)
        return 1

    messages = build_messages(DEFAULT_PROMPT, diff_input)

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer anythingshoulddo",
    }

    try:
        with requests.post(
            API_URL, headers=headers, json=payload, stream=True, timeout=600
        ) as resp:
            if resp.status_code >= 400:
                err = resp.json()
                msg = err.get("error", {}).get("message") or str(err)
                print(f"Error: {msg}", file=sys.stderr)
                return 1
            for chunk in stream_sse(resp):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()
    except requests.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
