#!/usr/bin/env -S python3
# Credit: https://old.reddit.com/r/LocalLLaMA/comments/1aim70j/embed_ollama_in_mac/
import asyncio
import json
import platform
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import requests
import sounddevice as sd
from kokoro_onnx import Kokoro


# CONFIG -- start


# Items in the drop-down. The keys appear in the drop-down and the values are sent in the prompt.
def get_chat_modes():
    return {
        "Summarise": "Provide summary in bullet points for the following text:",
        "Explain": "You are a writing assistant. Rewrite the text provided by the user to be more friendly. Output ONLY the friendly text without additional comments. Respond in the same language as the input (e.g., English US, French).",
        "Rewrite": "You are a writing assistant. Rewrite the text provided by the user to improve phrasing. Output ONLY the rewritten text without additional comments. Respond in the same language as the input (e.g., English US, French):",
        "Professional": "You are a writing assistant. Rewrite the text provided by the user to sound more professional. Output ONLY the professional text without additional comments. Respond in the same language as the input (e.g., English US, French):",
        "Friendly": "You are a writing assistant. Rewrite the text provided by the user to be more friendly. Output ONLY the friendly text without additional comments. Respond in the same language as the input (e.g., English US, French):",
        "Concise": "You are a writing assistant. Rewrite the text provided by the user to be slightly more concise in tone, thus making it just a bit shorter. Do not change the text too much or be too reductive. Output ONLY the concise version without additional comments. Respond in the same language as the input (e.g., English US, French):",
        "Proofread": "You are a grammar proofreading assistant. Output ONLY the corrected text without any additional comments. Maintain the original text structure and writing style. Respond in the same language as the input (e.g., English US, French):",
    }


# Ollama configuration and model
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_SELECTED_MODEL = "llama3.2:latest"

# Kokoro voice settings
MODEL_VOICES_PATH = Path.home() / "models/onnx/kokoro/voices.json"
MODEL_PATH = Path.home() / "models/onnx/kokoro/kokoro-v0_19.onnx"
MODEL_VOICE_SPEED = 1.0
MODEL_SELECTED_VOICE = "af_bella"

# Display text
TEXT_FONT = "Fantasque Sans Mono"
TEXT_FONT_SIZE = 18

# CONFIG -- end

try:
    import pyperclip

    USE_PYPERCLIP = True
except ImportError:
    USE_PYPERCLIP = False
    print("For better clipboard handling, install pyperclip: pip install pyperclip")


def create_popup():
    clipboard_content = None
    animation_running = False

    root = tk.Tk()
    root.withdraw()

    popup = tk.Toplevel(root)
    popup.overrideredirect(True)
    popup.config(bg="#FF8C00")

    outer_glow = tk.Frame(popup, bg="#FFA500", padx=3, pady=3)
    outer_glow.pack(fill="both", expand=True)

    inner_glow = tk.Frame(outer_glow, bg="#FFB84D", padx=2, pady=2)
    inner_glow.pack(fill="both", expand=True)

    main_frame = tk.Frame(inner_glow, bg="white")
    main_frame.pack(fill="both", expand=True)

    button_frame = tk.Frame(main_frame)
    button_frame.pack(side="top", fill="x", padx=10, pady=5)

    # Add speaking indicator
    speaking_indicator = tk.Canvas(
        button_frame,
        width=20,
        height=20,
        bg=button_frame.cget("bg"),
        highlightthickness=0,
    )
    indicator_circle = speaking_indicator.create_oval(
        5, 5, 15, 15, fill="gray", outline=""
    )

    content_frame = tk.Frame(main_frame, bg="white")
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
        font=(TEXT_FONT, TEXT_FONT_SIZE),
    )
    label.pack(fill="both", expand=True)

    canvas.create_window((0, 0), window=label_frame, anchor="nw")

    def pulse_animation():
        nonlocal animation_running
        if not animation_running:
            speaking_indicator.pack_forget()
            return

        current_color = speaking_indicator.itemcget(indicator_circle, "fill")
        new_color = "#4CAF50" if current_color == "gray" else "gray"
        speaking_indicator.itemconfig(indicator_circle, fill=new_color)
        popup.after(500, pulse_animation)

    async def start_speaking(text):
        nonlocal animation_running
        kokoro = Kokoro(
            model_path=MODEL_PATH.as_posix(), voices_path=MODEL_VOICES_PATH.as_posix()
        )
        stream = kokoro.create_stream(
            text,
            voice=MODEL_SELECTED_VOICE,
            speed=MODEL_VOICE_SPEED,
            lang="en-us",
        )

        # Start animation
        animation_running = True
        speaking_indicator.pack(side="left", padx=5)
        popup.after(0, pulse_animation)

        count = 0
        async for samples, sample_rate in stream:
            count += 1
            print(f"Playing audio stream ({count})...")
            sd.play(samples, sample_rate)
            sd.wait()

        # Stop animation
        animation_running = False
        speaking_indicator.itemconfig(indicator_circle, fill="gray")
        speaking_indicator.pack_forget()

    def make_api_call(text, update_callback):
        url = OLLAMA_ENDPOINT
        payload = {"model": OLLAMA_SELECTED_MODEL, "prompt": text}
        headers = {"Content-Type": "application/json"}
        complete_response = ""
        try:
            with requests.post(
                url, json=payload, headers=headers, stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        data = json.loads(decoded_line)
                        response_chunk = data.get("response", "")
                        update_callback(response_chunk)
                        complete_response = complete_response + response_chunk
                        if data.get("done", False):
                            break
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")

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

    def regenerate_response():
        label.config(text="")
        selected_mode = chat_mode_var.get()
        context = f"""{get_chat_modes()[selected_mode]}
        {clipboard_content}
        """
        threading.Thread(
            target=make_api_call, args=(context, update_message), daemon=True
        ).start()

    def speak_response():
        text_to_speak = label.cget("text")
        threading.Thread(
            target=lambda: asyncio.run(start_speaking(text_to_speak)), daemon=True
        ).start()

    close_button = tk.Button(
        button_frame,
        text="Close",
        command=close_popup,
    )
    close_button.pack(side="right", padx=(0, 0))

    copy_button = tk.Button(
        button_frame,
        text="Copy Text",
        command=copy_and_close,
    )
    copy_button.pack(side="right", padx=(0, 0))

    speak_button = tk.Button(
        button_frame,
        text="Speak",
        command=speak_response,
    )
    speak_button.pack(side="right", padx=(0, 5))

    chat_modes = get_chat_modes()
    chat_mode_var = tk.StringVar(value=list(chat_modes.keys())[0])
    chat_modes_dropdown = ttk.Combobox(
        button_frame,
        textvariable=chat_mode_var,
        values=list(chat_modes.keys()),
        state="readonly",
        width=15,
    )
    chat_modes_dropdown.pack(side="right", padx=(0, 5))
    chat_modes_dropdown.bind("<<ComboboxSelected>>", lambda e: regenerate_response())

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
    selected_mode = chat_mode_var.get()
    context = f"""{get_chat_modes()[selected_mode]}
    {clipboard_content}
    """
    threading.Thread(
        target=make_api_call, args=(context, update_message), daemon=True
    ).start()

    root.mainloop()


if __name__ == "__main__":
    create_popup()
