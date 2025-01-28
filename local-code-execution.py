#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "ipybox==0.3.1",
# ]
# ///
import asyncio

from ipybox import ExecutionClient
from ipybox import ExecutionContainer


async def main():
    async with ExecutionContainer(tag="ghcr.io/gradion-ai/ipybox:minimal") as container:
        async with ExecutionClient(port=container.port) as client:
            result = await client.execute("print('Hello, world!')")
            print(f"Output: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
