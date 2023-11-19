from openai import OpenAI

client = OpenAI()
import typer
from rich import print
from rich.table import Table


def main():
    print("💬 [bold green]ChatGPT API en Python[/bold green]")

    table = Table("Command", "Description")
    table.add_row("exit", "Exit the application")
    table.add_row("new", "Create new conversation")

    print(table)

    context = {"role": "system", "content": "You are a very helpful assistant."}
    messages = [context]

    while True:
        content = __prompt()

        if content == "new":
            print("🆕 New conversation created")
            messages = [context]
            content = __prompt()

        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

        response_content = response.choices[0].message.content

        messages.append({"role": "assistant", "content": response_content})

        print(f"[bold green]> [/bold green] [green]{response_content}[/green]")


def __prompt() -> str:
    prompt = typer.prompt("\nWhat do you want to talk about?")

    if prompt == "exit":
        exit = typer.confirm("✋ Are you sure?")
        if exit:
            print("👋 See you later!")
            raise typer.Abort()

        return __prompt()

    return prompt


if __name__ == "__main__":
    typer.run(main)
