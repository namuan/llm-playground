#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "vlmrun",
#   "ollama",
# ]
# ///
from ollama import chat

from vlmrun.common.image import encode_image
from vlmrun.common.utils import remote_image
from vlmrun.hub.schemas.document.invoice import Invoice


IMAGE_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.invoice/invoice_1.jpg"

img = remote_image(IMAGE_URL)
chat_response = chat(
    model="llama3.2-vision:latest",
    format=Invoice.model_json_schema(),
    messages=[
        {
            "role": "user",
            "content": "Extract the invoice in JSON.",
            "images": [encode_image(img, format="JPEG").split(",")[1]],
        },
    ],
    options={
        "temperature": 0
    },
)
response = Invoice.model_validate_json(
    chat_response.message.content
)
print(response)
