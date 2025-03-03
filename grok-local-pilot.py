#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "starlette",
#   "uvicorn"
# ]
# ///
"""
OpenAI API Mock - A simple mock server that simulates OpenAI API responses

Usage:
./grok-local-pilot.py -h

./grok-local-pilot.py -v # To log INFO messages
./grok-local-pilot.py -vv # To log DEBUG messages

This script runs a mock server that returns static responses for OpenAI API requests.
Example request:

curl http://localhost:5001/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
     "model": "gpt-4o-mini",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
"""
import logging
import json
import time
import uuid
import uvicorn
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from starlette import applications, responses
from starlette.requests import Request
from starlette.routing import Route


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the mock server on (default: 5001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    return parser.parse_args()


async def mock_chat_completions(request: Request):
    """Handle chat completions requests and return a static response."""
    try:
        # Get the request body
        body_bytes = await request.body()
        if body_bytes:
            try:
                # Parse the request to extract useful information
                request_data = json.loads(body_bytes)
                model = request_data.get("model", "gpt-4o-mini")
                messages = request_data.get("messages", [])

                # Log the request details
                logging.info(f"Received request for model: {model}")
                if messages:
                    last_message = messages[-1].get("content", "") if messages else ""
                    logging.info(f"Last message content: {last_message[:50]}...")

            except json.JSONDecodeError:
                logging.warning("Could not parse request body as JSON")
                request_data = {}
                model = "gpt-4o-mini"
        else:
            request_data = {}
            model = "gpt-4o-mini"

        # Create a dummy response
        current_time = int(time.time())
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": current_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test! I'm a mock response from your local proxy server."
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 14,
                "total_tokens": 26
            },
            "system_fingerprint": "fp_" + uuid.uuid4().hex[:16]
        }

        # Simulate a slight delay to mimic API response time
        await asyncio.sleep(0.5)

        logging.info("Returning mock response")
        return responses.JSONResponse(response_data)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return responses.JSONResponse(
            {"error": {"message": f"Error: {str(e)}", "type": "server_error"}},
            status_code=500
        )


async def handle_root(request: Request):
    """Simple handler for the root path to confirm the server is running."""
    return responses.JSONResponse({
        "status": "running",
        "message": "OpenAI API mock server is running. Send requests to /v1/chat/completions"
    })


async def handle_404(request: Request, exc):
    """Handle 404 errors."""
    return responses.JSONResponse({
        "error": {
            "message": "Not found. Use /v1/chat/completions for chat completions.",
            "type": "invalid_request_error",
            "param": None,
            "code": "resource_not_found"
        }
    }, status_code=404)


async def handle_500(request: Request, exc):
    """Handle 500 errors."""
    logging.error(f"Server error: {exc}")
    return responses.JSONResponse({
        "error": {
            "message": f"Server error: {str(exc)}",
            "type": "server_error",
            "param": None,
            "code": "internal_server_error"
        }
    }, status_code=500)


def run_mock_server(host, port):
    """Run the mock server."""
    # Define routes
    routes = [
        Route("/", handle_root),
        Route("/v1/chat/completions", mock_chat_completions, methods=["POST"]),
    ]

    # Create app with routes
    app = applications.Starlette(
        routes=routes,
        exception_handlers={
            404: handle_404,
            500: handle_500
        }
    )

    logging.info(f"Starting OpenAI API mock server on {host}:{port}")
    logging.info(f"Send requests to http://{host}:{port}/v1/chat/completions")

    uvicorn.run(app, host=host, port=port)


def main(args):
    # Run the mock server
    run_mock_server(args.host, args.port)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    import asyncio  # Import here to avoid issues with top-level async
    main(args)