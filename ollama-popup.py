# Credit: https://old.reddit.com/r/LocalLLaMA/comments/1aim70j/embed_ollama_in_mac/

import json
import sys
import threading
import tkinter as tk

import requests


def make_api_call(text, update_callback):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "mistral:latest", "prompt": text, "stream": True}
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    data = json.loads(decoded_line)
                    update_callback(data.get("response", ""))
                    if data.get("done", False):
                        break
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")


def create_popup():
    root = tk.Tk()
    root.withdraw()

    popup = tk.Toplevel(root)
    popup.overrideredirect(True)
    popup.config(bg="white")

    label = tk.Label(
        popup,
        text="",
        bg="white",
        anchor="nw",
        justify="left",
        wraplength=680,
    )
    label.pack(padx=10, pady=10)

    def update_message(message):
        current_text = label.cget("text")
        label.config(text=current_text + message)
        popup.update_idletasks()
        window_height = label.winfo_reqheight() + 20
        window_width = 700
        x_position = int(popup.winfo_screenwidth() / 2 - window_width / 2)
        y_position = int(popup.winfo_screenheight() / 8)
        popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        popup.lift()

    def on_click(event):
        popup.destroy()
        root.destroy()

    popup.bind("<Button-1>", on_click)

    clipboard_content = sys.argv[1] if len(sys.argv) > 1 else "An error occurred"
    update_message("")
    threading.Thread(
        target=make_api_call, args=(clipboard_content, update_message), daemon=True
    ).start()

    popup.after(30000, lambda: on_click(None))
    root.mainloop()


if __name__ == "__main__":
    create_popup()
