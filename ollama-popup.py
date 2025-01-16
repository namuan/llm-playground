#!/usr/bin/env -S python3
# Credit: https://old.reddit.com/r/LocalLLaMA/comments/1aim70j/embed_ollama_in_mac/

import json
import platform
import threading
import time
import tkinter as tk
from tkinter import ttk

import requests

try:
    import pyperclip

    USE_PYPERCLIP = True
except ImportError:
    USE_PYPERCLIP = False
    print("For better clipboard handling, install pyperclip: pip install pyperclip")


def get_chat_mode():
    return ["Explain", "Summarise", "Proofread"]


def make_api_call(text, update_callback):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3.2:latest", "prompt": text}
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

    button_frame = tk.Frame(popup)
    button_frame.pack(side="top", fill="x", padx=10, pady=5)

    content_frame = tk.Frame(popup, bg="white")
    content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

    canvas = tk.Canvas(content_frame, bg="white", highlightthickness=0)
    scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)

    label_frame = tk.Frame(canvas, bg="white")

    label = tk.Label(
        label_frame,
        text="",
        bg="white",
        anchor="nw",
        justify="left",
        wraplength=660,
    )
    label.pack(fill="both", expand=True)

    canvas.create_window((0, 0), window=label_frame, anchor="nw")

    def copy_and_close():
        text_to_copy = label.cget("text")

        if USE_PYPERCLIP:
            try:
                pyperclip.copy(text_to_copy)
            except Exception as e:
                print(f"Pyperclip error: {e}")
                popup.clipboard_clear()
                popup.clipboard_append(text_to_copy)
                popup.update()
        else:
            popup.clipboard_clear()
            popup.clipboard_append(text_to_copy)
            popup.update()

        time.sleep(0.5)
        popup.destroy()
        root.destroy()

    def close_popup():
        popup.destroy()
        root.destroy()

    def regenerate_response():
        label.config(text="")
        try:
            if USE_PYPERCLIP:
                clipboard_content = pyperclip.paste()
            else:
                clipboard_content = root.clipboard_get()
        except Exception as e:
            clipboard_content = "No text found in clipboard"
            print(f"Clipboard error: {e}")

        selected_mode = model_var.get()
        context = f"""
        {selected_mode}
        {clipboard_content}
        """
        threading.Thread(
            target=make_api_call, args=(context, update_message), daemon=True
        ).start()

    close_button = tk.Button(
        button_frame,
        text="Close",
        command=close_popup,
    )
    close_button.pack(side="right", padx=(0, 5))

    copy_button = tk.Button(
        button_frame,
        text="Copy Text",
        command=copy_and_close,
    )
    copy_button.pack(side="right", padx=(0, 5))

    # Add mode selection dropdown
    chat_modes = get_chat_mode()
    model_var = tk.StringVar(value=chat_modes[0])
    chat_modes_dropdown = ttk.Combobox(
        button_frame,
        textvariable=model_var,
        values=chat_modes,
        state="readonly",
        width=15,
    )
    chat_modes_dropdown.pack(side="right", padx=(0, 5))

    # Add regenerate button
    regenerate_button = tk.Button(
        button_frame,
        text="Regenerate",
        command=regenerate_response,
    )
    regenerate_button.pack(side="right", padx=(0, 5))

    def update_message(message):
        current_text = label.cget("text")
        label.config(text=current_text + message)

        label_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        content_height = label_frame.winfo_reqheight()
        max_height = min(
            content_height + button_frame.winfo_reqheight() + 30,
            popup.winfo_screenheight() // 2,
        )
        window_width = 700

        if content_height > (max_height - button_frame.winfo_reqheight() - 30):
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
        else:
            scrollbar.pack_forget()
            canvas.pack(side="left", fill="both", expand=True)

        x_position = int(popup.winfo_screenwidth() / 2 - window_width / 2)
        y_position = int(popup.winfo_screenheight() / 8)
        popup.geometry(f"{window_width}x{max_height}+{x_position}+{y_position}")

        canvas.configure(height=max_height - button_frame.winfo_reqheight() - 30)
        canvas.configure(yscrollcommand=scrollbar.set)
        popup.lift()

    def on_scroll(event):
        if platform.system() == "Darwin":
            canvas.yview_scroll(int(event.delta), "units")
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_scroll)
    canvas.bind_all("<Shift-MouseWheel>", on_scroll)
    canvas.bind_all("<Control-MouseWheel>", on_scroll)

    try:
        if USE_PYPERCLIP:
            clipboard_content = pyperclip.paste()
        else:
            clipboard_content = root.clipboard_get()
    except Exception as e:
        clipboard_content = "No text found in clipboard"
        print(f"Clipboard error: {e}")

    update_message("")
    selected_mode = model_var.get()
    context = f"""
    {selected_mode}
    {clipboard_content}
    """
    threading.Thread(
        target=make_api_call, args=(context, update_message), daemon=True
    ).start()

    root.mainloop()


if __name__ == "__main__":
    create_popup()
