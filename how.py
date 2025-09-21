#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests"
# ]
# ///

import argparse
import json
import logging
import sys
from typing import Iterable, Optional

import requests

logger = logging.getLogger("how")

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

DEFAULT_PROMPT = (
    "You are a direct answer engine. Output ONLY the requested information.\n\n"
    "For commands: Output executable syntax only. No explanations, no comments.\n"
    "For questions: Output the answer only. No context, no elaboration.\n\n"
    "Rules:\n"
    "- If asked for a command, provide ONLY the command\n"
    "- If asked a question, provide ONLY the answer\n"
    "- Never include markdown formatting or code blocks\n"
    "- Never add explanatory text before or after\n"
    "- Assume output will be piped or executed directly\n"
    "- For multi-step commands, use && or ; to chain them\n"
    "- Make commands robust and handle edge cases silently"
)


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(
        prog="ask",
        description="ask - Query AI models via OpenRouter API",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Only supported flags
    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message")

    # Capture remaining as single PROMPT positional like the shell script
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    # Join remaining prompt tokens into a single string (like "$*")
    prompt_from_args = " ".join(args.prompt).strip() if args.prompt else ""

    return {
        "model": "any-model-should-work",
        "prompt_from_args": prompt_from_args,
        "verbose": int(args.verbose or 0),
    }


def read_prompt(prompt_from_args: str) -> str:
    if prompt_from_args:
        return "how " + prompt_from_args

    print("Error: No prompt provided. Use 'ask -h' for help.", file=sys.stderr)
    sys.exit(1)


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


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = parse_args(argv)

    verbose = int(cfg.get("verbose", 0) or 0)
    debug_enabled = verbose >= 2
    # Configure logging: only emit DEBUG when -vv; -v sets INFO; default WARNING
    level = logging.WARNING
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    if debug_enabled:
        try:
            argv_list = list(argv) if argv is not None else sys.argv[1:]
        except Exception:
            argv_list = []
        logger.debug("argv=%s", argv_list)
        logger.debug("config: model=%s, stream=True, verbose=%s", cfg["model"], verbose)

    user_prompt = read_prompt(cfg["prompt_from_args"])

    if debug_enabled:
        logger.debug("prompt length=%d chars", len(user_prompt))

    messages = build_messages(DEFAULT_PROMPT, user_prompt)

    payload = {
        "model": cfg["model"],
        "messages": messages,
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer anythingshoulddo",
    }

    if debug_enabled:
        redacted_headers = dict(headers)
        if "Authorization" in redacted_headers:
            redacted_headers["Authorization"] = "Bearer ****"
        logger.debug("POST %s", API_URL)
        logger.debug("headers=%s", redacted_headers)
        try:
            # Do not dump full messages to avoid huge logs; show roles and sizes
            msg_info = [
                {"role": m.get("role"), "len": len(str(m.get("content", "")))}
                for m in payload.get("messages", [])
            ]
            logger.debug("payload keys=%s, messages=%s", list(payload.keys()), msg_info)
        except Exception as e:
            logger.debug("payload debug error: %s", e)

    print()

    try:
        with requests.post(
            API_URL, headers=headers, json=payload, stream=True, timeout=600
        ) as resp:
            if debug_enabled:
                logger.debug("response status=%s", resp.status_code)
                logger.debug(
                    "response content-type=%s", resp.headers.get("Content-Type", "")
                )
            # If server sends JSON error first, try to parse
            ct = resp.headers.get("Content-Type", "")
            if resp.status_code >= 400 or (
                "application/json" in ct and not ct.startswith("text/event-stream")
            ):
                try:
                    err = resp.json()
                    msg = (
                        err.get("error", {}).get("message")
                        or err.get("error")
                        or str(err)
                    )
                    if debug_enabled:
                        logger.debug("error json=%s", err)
                except Exception:
                    msg = resp.text
                    if debug_enabled:
                        preview = (msg or "")[:500]
                        logger.debug("error body preview=%s", preview)
                print(f"Error: {msg}", file=sys.stderr)
                return 1
            if debug_enabled:
                logger.debug("starting SSE stream...")
            for chunk in stream_sse(resp, verbose=verbose):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()
    except requests.RequestException as e:
        if debug_enabled:
            logger.debug("request exception type=%s", type(e).__name__)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
