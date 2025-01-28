#!uv run
# /// script
# dependencies = [
#   "timm",
#   "transformers",
#   "einops",
# ]
# ///
import argparse

import torch
from tinyclick_utils import postprocess
from tinyclick_utils import prepare_inputs
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(
        prog="TinyClickInference",
        description="Example inference with TinyClick agent.",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        type=str,
        help="Path to the input image (GUI screenshot).",
    )
    parser.add_argument(
        "--text",
        required=True,
        type=str,
        help="Input command to perform by agent on GUI screenshot.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        "Samsung/TinyClick", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Samsung/TinyClick",
        trust_remote_code=True,
    ).to(device)

    text = args.text
    image_path = args.image_path

    inputs = prepare_inputs(image_path, text, processor)
    img_size = inputs.pop("image_size")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs)

    generated_texts = processor.batch_decode(outputs, skip_special_tokens=False)

    result = postprocess(generated_texts[0], img_size)
    print(result)


if __name__ == "__main__":
    main()
