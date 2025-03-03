#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "starlette",
#   "uvicorn",
#   "requests",
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
import json
import logging
import os
import time
import uuid
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import requests
import uvicorn
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
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
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
        help="Port to run the mock server on (default: 5001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    return parser.parse_args()


class GrokClient:
    def __init__(self, cookies):
        """
        Initialize the Grok client with cookie values

        Args:
            cookies (dict): Dictionary containing cookie values
                - x-anonuserid
                - x-challenge
                - x-signature
                - sso
                - sso-rw
        """
        self.base_url = "https://grok.com/rest/app-chat/conversations/new"
        self.cookies = cookies
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }

    def _prepare_payload(self, message):
        """Prepare the default payload with the user's message"""
        return {
            "temporary": False,
            "modelName": "grok-3",
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": False,
        }

    def send_message(self, message):
        """
        Send a message to Grok and collect the streaming response

        Args:
            message (str): The user's input message

        Returns:
            str: The complete response from Grok
        """
        payload = self._prepare_payload(message)
        response = requests.post(
            self.base_url,
            headers=self.headers,
            cookies=self.cookies,
            json=payload,
            stream=True,
        )

        full_response = ""

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                try:
                    json_data = json.loads(decoded_line)
                    result = json_data.get("result", {})
                    response_data = result.get("response", {})

                    if "modelResponse" in response_data:
                        response_message = response_data["modelResponse"]["message"]
                        logging.info(f"Complete Grok response: {response_message}")
                        return response_message

                    token = response_data.get("token", "")
                    if token:
                        full_response += token
                        logging.debug(f"Received token: {token}")

                except json.JSONDecodeError:
                    continue

        logging.info(f"Complete Grok response: {full_response.strip()}")
        return full_response.strip()


def load_grok_cookies():
    """Load Grok cookies from .grok.env file"""
    cookies = {}
    try:
        # Read .grok.env file
        if os.path.exists(".grok.env"):
            with open(".grok.env", "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        cookies[key.strip()] = value.strip().strip("\"'")

            logging.info(f"Successfully loaded Grok cookies: {list(cookies.keys())}")
            return cookies
        else:
            logging.error(".grok.env file not found")
            return None
    except Exception as e:
        logging.error(f"Error loading Grok cookies: {e}")
        return None


grok_client = None


async def mock_chat_completions(request: Request):
    """Handle chat completions requests and forward to Grok."""
    global grok_client

    try:
        # Get the request body
        body_bytes = await request.body()
        if not body_bytes:
            return responses.JSONResponse(
                {
                    "error": {
                        "message": "No request body provided",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )

        try:
            # Parse the request
            request_data = json.loads(body_bytes)
            model = request_data.get("model", "gpt-4o-mini")
            messages = request_data.get("messages", [])

            if not messages:
                return responses.JSONResponse(
                    {
                        "error": {
                            "message": "No messages provided",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )

            # Extract the user's message
            last_message = messages[-1].get("content", "")
            logging.info(
                f"Received request - Model: {model}, Message: {last_message[:50]}..."
            )

            # Initialize Grok client if needed
            if grok_client is None:
                cookies = load_grok_cookies()
                if cookies:
                    grok_client = GrokClient(cookies)
                    logging.info("Initialized Grok client")
                else:
                    return responses.JSONResponse(
                        {
                            "error": {
                                "message": "Could not load Grok cookies",
                                "type": "server_error",
                            }
                        },
                        status_code=500,
                    )

            # Send the message to Grok
            logging.info(f"Forwarding request to Grok: {last_message[:100]}...")
            grok_response = grok_client.send_message(last_message)

            # Print full response for debugging
            print("\n---------- GROK RESPONSE ----------")
            print(grok_response)
            print("-----------------------------------\n")

            # Create a response in OpenAI format with the actual Grok response
            current_time = int(time.time())
            response_data = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": current_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": grok_response},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(last_message.split()) * 2,
                    "completion_tokens": len(grok_response.split()) * 2,
                    "total_tokens": (
                        len(last_message.split()) + len(grok_response.split())
                    )
                    * 2,
                },
                "system_fingerprint": "fp_" + uuid.uuid4().hex[:16],
            }

            logging.info(f"Returning Grok response (length: {len(grok_response)})")
            return responses.JSONResponse(response_data)

        except json.JSONDecodeError:
            logging.warning("Could not parse request body as JSON")
            return responses.JSONResponse(
                {
                    "error": {
                        "message": "Could not parse request body as JSON",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return responses.JSONResponse(
            {"error": {"message": f"Error: {str(e)}", "type": "server_error"}},
            status_code=500,
        )


async def handle_root(request: Request):
    """Simple handler for the root path to confirm the server is running."""
    return responses.JSONResponse(
        {
            "status": "running",
            "message": "Grok proxy server is running. Send requests to /v1/chat/completions",
            "grok_client_ready": grok_client is not None,
        }
    )


async def handle_404(request: Request, exc):
    """Handle 404 errors."""
    return responses.JSONResponse(
        {
            "error": {
                "message": "Not found. Use /v1/chat/completions for chat completions.",
                "type": "invalid_request_error",
                "param": None,
                "code": "resource_not_found",
            }
        },
        status_code=404,
    )


async def handle_500(request: Request, exc):
    """Handle 500 errors."""
    logging.error(f"Server error: {exc}")
    return responses.JSONResponse(
        {
            "error": {
                "message": f"Server error: {str(exc)}",
                "type": "server_error",
                "param": None,
                "code": "internal_server_error",
            }
        },
        status_code=500,
    )


def run_mock_server(host, port):
    """Run the mock server."""
    # Define routes
    routes = [
        Route("/", handle_root),
        Route("/v1/chat/completions", mock_chat_completions, methods=["POST"]),
    ]

    # Create app with routes
    app = applications.Starlette(
        routes=routes, exception_handlers={404: handle_404, 500: handle_500}
    )

    # Try to initialize Grok client at startup
    global grok_client
    cookies = load_grok_cookies()
    if cookies:
        grok_client = GrokClient(cookies)
        logging.info("Initialized Grok client at startup")
    else:
        logging.warning(
            "Could not initialize Grok client at startup - will try during first request"
        )

    logging.info(f"Starting Grok proxy server on {host}:{port}")
    logging.info(f"Send requests to http://{host}:{port}/v1/chat/completions")

    uvicorn.run(app, host=host, port=port)


def main(args):
    # Run the mock server
    run_mock_server(args.host, args.port)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main(args)
