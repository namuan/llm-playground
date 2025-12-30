#!/usr/bin/env -S python3
import asyncio
import platform
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import sounddevice as sd
from kokoro_onnx import Kokoro


MODEL_VOICES_PATH = Path.home() / "models/onnx/kokoro/voices.json"
MODEL_PATH = Path.home() / "models/onnx/kokoro/kokoro-v0_19.onnx"
MODEL_VOICE_SPEED = 1.0
MODEL_SELECTED_VOICE = "af_bella"

TEXT_FONT = "Fantasque Sans Mono"
TEXT_FONT_SIZE = 18


def create_popup():
    animation_running = False
    speaking_thread = None
    stop_playback_event = threading.Event()
    playback_loop = None
    playback_stream = None

    root = tk.Tk()
    root.withdraw()

    popup = tk.Toplevel(root)
    popup.overrideredirect(False)
    popup.config(bg="#FF8C00")
    popup.resizable(True, True)
    popup.title("Kokoro - Text to Speech")

    outer_glow = tk.Frame(popup, bg="#FFA500", padx=3, pady=3)
    outer_glow.pack(fill="both", expand=True)

    inner_glow = tk.Frame(outer_glow, bg="#FFB84D", padx=2, pady=2)
    inner_glow.pack(fill="both", expand=True)

    main_frame = tk.Frame(inner_glow, bg="white")
    main_frame.pack(fill="both", expand=True)

    button_frame = tk.Frame(main_frame)
    button_frame.pack(side="top", fill="x", padx=10, pady=5)

    speaking_indicator = tk.Canvas(
        button_frame, width=20, height=20, bg=button_frame.cget("bg"), highlightthickness=0
    )
    indicator_circle = speaking_indicator.create_oval(5, 5, 15, 15, fill="gray", outline="")

    content_frame = tk.Frame(main_frame, bg="white")
    content_frame.pack(side="top", fill="both", expand=True, padx=0, pady=0)

    text_area = tk.Text(
        content_frame,
        wrap="word",
        font=(TEXT_FONT, TEXT_FONT_SIZE),
        bg="white",
        relief="flat",
        highlightthickness=0,
    )
    scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=text_area.yview)
    text_area.configure(yscrollcommand=scrollbar.set)
    text_area.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def pulse_animation():
        nonlocal animation_running
        if not animation_running or not popup.winfo_exists():
            speaking_indicator.pack_forget()
            return
        try:
            current_color = speaking_indicator.itemcget(indicator_circle, "fill")
            new_color = "#4CAF50" if current_color == "gray" else "gray"
            speaking_indicator.itemconfig(indicator_circle, fill=new_color)
        except Exception:
            return
        popup.after(500, pulse_animation)

    async def start_speaking(text, stop_event):
        nonlocal animation_running, playback_stream
        kokoro = Kokoro(model_path=MODEL_PATH.as_posix(), voices_path=MODEL_VOICES_PATH.as_posix())
        stream = kokoro.create_stream(text, voice=MODEL_SELECTED_VOICE, speed=MODEL_VOICE_SPEED, lang="en-us")
        playback_stream = stream
        animation_running = True
        speaking_indicator.pack(side="left", padx=5)
        popup.after(0, pulse_animation)
        try:
            async for samples, sample_rate in stream:
                if stop_event.is_set():
                    try:
                        sd.stop()
                    except Exception:
                        pass
                    break
                try:
                    sd.play(samples, sample_rate)
                    sd.wait()
                except sd.PortAudioError:
                    break
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            try:
                sd.stop()
            except Exception:
                pass
            try:
                await stream.aclose()
            except Exception:
                pass
            playback_stream = None
            animation_running = False
            if popup.winfo_exists():
                try:
                    speaking_indicator.itemconfig(indicator_circle, fill="gray")
                    speaking_indicator.pack_forget()
                except Exception:
                    pass

    def speak_response():
        nonlocal speaking_thread, playback_loop
        text_to_speak = text_area.get("1.0", "end").strip()
        if not text_to_speak:
            return
        stop_playback_event.clear()
        def runner():
            nonlocal playback_loop
            playback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(playback_loop)
            try:
                playback_loop.run_until_complete(start_speaking(text_to_speak, stop_playback_event))
            finally:
                try:
                    pending = [t for t in asyncio.all_tasks() if not t.done()]
                    if pending:
                        for t in pending:
                            try:
                                t.cancel()
                            except Exception:
                                pass
                        try:
                            playback_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        except Exception:
                            pass
                except Exception:
                    pass
                playback_loop.close()
                playback_loop = None
        speaking_thread = threading.Thread(target=runner, daemon=True)
        speaking_thread.start()

    def close_popup():
        nonlocal speaking_thread, playback_loop, playback_stream, animation_running
        stop_playback_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        try:
            if playback_stream and playback_loop:
                fut = asyncio.run_coroutine_threadsafe(playback_stream.aclose(), playback_loop)
                try:
                    fut.result(timeout=1.0)
                except Exception:
                    pass
        except Exception:
            pass
        if speaking_thread and speaking_thread.is_alive():
            try:
                speaking_thread.join(timeout=1.5)
            except Exception:
                pass
        animation_running = False
        popup.destroy()
        root.destroy()
    popup.protocol("WM_DELETE_WINDOW", close_popup)

    close_button = tk.Button(button_frame, text="Close", command=close_popup)
    close_button.pack(side="right", padx=(0, 0))

    speak_button = tk.Button(button_frame, text="Speak", command=speak_response)
    speak_button.pack(side="right", padx=(0, 5))

    window_width = 700
    window_height = 400
    x_position = int(popup.winfo_screenwidth() / 2 - window_width / 2)
    y_position = int(popup.winfo_screenheight() / 8)
    popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    popup.lift()

    def on_scroll(event):
        if platform.system() == "Darwin":
            text_area.yview_scroll(int(event.delta), "units")
        else:
            text_area.yview_scroll(int(-1 * (event.delta / 120)), "units")

    text_area.bind_all("<MouseWheel>", on_scroll)
    text_area.bind_all("<Shift-MouseWheel>", on_scroll)
    text_area.bind_all("<Control-MouseWheel>", on_scroll)

    root.mainloop()


if __name__ == "__main__":
    create_popup()
