#!/usr/bin/env python3
"""
A script that analyzes screenshots using a local LLM (qwen2-vl-2b-instruct) via LM Studio API.

The script takes a screenshot image, encodes it to base64, and sends it along with a prompt
to the LLM via the OpenAI-compatible API. The LLM analyzes the image content and returns
structured information about visible applications and overall screen content, validated against
a Pydantic schema.

Usage:
./lm-studio-describe-screenshot.py -h

Dependencies:
- LM Studio > 0.3.5 (Build 2)
  * mlx-community/Qwen2-VL-2B-Instruct-4bit
- openai
- Pillow
- pydantic
"""

import base64
import io
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
from typing import Annotated
from typing import List

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from pydantic import Field
from pydantic.types import StringConstraints


def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    image = Image.open(
        Path.home() / "Documents" / "Screenshots/2024/10/14/20241014_083302.png"
    )
    # Convert PIL Image to RGB mode
    image_rgb = image.convert("RGB")

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image_rgb.save(img_byte_arr, format="JPEG")
    image_bytes = img_byte_arr.getvalue()

    # Encode to base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    class RunningApplication(BaseModel):
        name: Annotated[str, StringConstraints(min_length=3)] = Field(
            description="Name of the running application"
        )
        summary: str = Field(description="What is being displayed in the application")

    class ScreenshotInformation(BaseModel):
        description: Annotated[
            str, StringConstraints(min_length=50, max_length=1000)
        ] = Field(
            ...,
            description="A brief summary of the overall screen content and any patterns or notable observations across the applications.",
        )
        applications: List[RunningApplication] = Field(
            [], description="A list of applications visible on the screen"
        )

    prompt = f"""Please examine the image I've provided and provide the following information in JSON format:

1. description: A brief summary of the overall screen content and any patterns or notable observations across the applications.
2. applications: List of running applications with the following information:
    a. Identify the application name or type.
    b. Describe what's happening or being displayed in that application.
    c. Note any significant details or status indicators (e.g., open tabs, notifications, etc.).

Ensure your response follows this schema:
{ScreenshotInformation.model_json_schema()}

Do not include any explanations or additional text outside of the JSON structure."""

    completion = client.beta.chat.completions.parse(
        model="qwen2-vl-2b-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        response_format=ScreenshotInformation,
    )

    print(completion.choices[0].message.parsed)


if __name__ == "__main__":
    args = parse_args()
    main(args)
