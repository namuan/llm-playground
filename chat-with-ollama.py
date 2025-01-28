#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "gradio",
#   "langchain",
#   "langchain-community",
#   "langchain-ollama",
#   "openai",
#   "pypdf",
#   "chromadb",
#   "youtube-transcript-api",
# ]
# ///
import argparse
import os

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

from providers.ollama import OllamaProvider

os.environ["TOKENIZERS_PARALLELISM"] = "False"
rag_prompt_default = """You are given a context from a document and your job is to answer a question from a user about that given context
---CONTEXT---
{context}
---END---
Based on the given context and information. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
AI Response:
"""

rag_prompt_history_default = """
You are given a context from a document and a chat history between the user and you. Your job is to answer a question from a user about that given context and the chat history:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Based on the given context, information and chat history. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
AI Response:
"""

openai_api_base = "http://127.0.0.1:11434/v1"
# model_dicts, yml_path, cfg_list, mlx_config = model_info()
# model_list = list(cfg_list.keys())
client = OpenAI(api_key="EMPTY", base_url=openai_api_base)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
emb = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = None
models_provider = OllamaProvider()
model_list = models_provider.get_available_models()


def check_file_type(file_path):
    # Check for document file extensions
    if (
        file_path.endswith(".pdf")
        or file_path.endswith(".txt")
        or file_path.endswith(".doc")
        or file_path.endswith(".docx")
    ):
        return True
        # Check for YouTube link formats
    elif (
        file_path.startswith("https://www.youtube.com/")
        or file_path.startswith("https://youtube.com/")
        or file_path.startswith("https://youtu.be/")
    ):
        return True
    else:
        return False


def upload(files):
    supported = check_file_type(files)
    if supported:
        return {url: files, index_status: "Not Done"}
    else:
        return {url: "File type not supported", index_status: "Not Done"}


def indexing(mode, url):
    global vectorstore

    try:
        if mode == "Files (docx, pdf, txt)":
            if url.endswith(".pdf"):
                loader = PyPDFLoader(url)
            elif url.endswith(".docx"):
                loader = Docx2txtLoader(url)
            elif url.endswith(".txt"):
                loader = TextLoader(url)
            splits = loader.load_and_split(text_splitter)
        elif mode == "YouTube (url)":
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=False, language=["en", "vi"]
            )
            splits = loader.load_and_split(text_splitter)

        vectorstore = Chroma.from_documents(documents=splits, embedding=emb)
        return {index_status: "Indexing Done"}
    except Exception as e:
        # Print the error message or return it as part of the response
        print(f"Error: {e}")  # This will print the error to the console or log
        return {"index_status": "Indexing Error", "error_message": str(e)}


def kill_index():
    global vectorstore
    vectorstore = None
    return {index_status: "Indexing Undone"}


def build_rag_context(docs):
    context = ""
    for doc in docs:
        context += doc.page_content + "\n"

    return context


def chatbot(query, history, model_name, temp, max_tokens, freq_penalty, k_docs):
    global chat_history, sys_prompt

    if "vectorstore" in globals() and vectorstore is not None:
        if len(history) == 0:
            chat_history = []
            docs = vectorstore.similarity_search(query, k=k_docs)
        else:
            history_str = ""
            for i, message in enumerate(history):
                history_str += f"User: {message[0]}\n"
                history_str += f"AI: {message[1]}\n"

            chat_history.append({"role": "user", "content": history_str})
            docs = vectorstore.similarity_search(history_str)

        context = build_rag_context(docs)

        if len(history) == 0:
            prompt = rag_prompt_default.format(context=context, question=query)
        else:
            prompt = rag_prompt_history_default.format(
                chat_history=history_str, context=context, question=query
            )
        messages = [{"role": "user", "content": prompt}]
    else:
        chat_history = []
        for i, message in enumerate(history):
            chat_history.append({"role": "user", "content": message[0]})
            chat_history.append({"role": "assistant", "content": message[1]})
        chat_history.append({"role": "user", "content": query})
        messages = chat_history

    # Uncomment for debugging
    print(messages)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temp,
        frequency_penalty=freq_penalty,
        max_tokens=max_tokens,
        stream=True,
    )
    stop = ["<|im_end|>", "<|endoftext|>"]
    partial_message = ""
    for chunk in response:
        if len(chunk.choices) != 0:
            if chunk.choices[0].delta.content not in stop:
                partial_message = partial_message + chunk.choices[0].delta.content
            else:
                partial_message = partial_message + ""
            yield partial_message


with gr.Blocks(fill_height=True, theme=gr.themes.Soft()) as demo:
    model_name = gr.Dropdown(
        label="Model",
        info="Select your model",
        choices=sorted(model_list),
        value=model_list[0],
        render=False,
    )
    temp_slider = gr.State(0.2)
    max_gen_token = gr.State(512)
    freq_penalty = gr.State(1.05)
    retrieve_docs = gr.State(3)
    gr.ChatInterface(
        chatbot=gr.Chatbot(height=600, render=False),
        fn=chatbot,  # Function to call on user input
        title="Chat with Ollama🍎",  # Title of the web page
        additional_inputs=[
            model_name,
            temp_slider,
            max_gen_token,
            freq_penalty,
            retrieve_docs,
        ],
    )
    with gr.Accordion("Advanced Setting", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                temp_slider = gr.Slider(
                    label="Temperature",
                    value=0.2,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    interactive=True,
                )
                max_gen_token = gr.Slider(
                    label="Max Tokens",
                    value=512,
                    minimum=512,
                    maximum=4096,
                    step=256,
                    interactive=True,
                )
            with gr.Column(scale=2):
                freq_penalty = gr.Slider(
                    label="Frequency Penalty",
                    value=1.05,
                    minimum=-2,
                    maximum=2,
                    step=0.05,
                    interactive=True,
                )
                retrieve_docs = gr.Slider(
                    label="No. Retrieval Docs",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1,
                    interactive=True,
                )

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=9):
                    mode = gr.Dropdown(
                        label="Dataset",
                        info="Choose your dataset type",
                        choices=["Files (docx, pdf, txt)", "YouTube (url)"],
                        scale=5,
                    )
                    url = gr.Textbox(
                        label="URL",
                        info="Enter your filepath (URL for Youtube)",
                        interactive=True,
                    )
                    upload_button = gr.UploadButton(
                        label="Upload File", variant="primary"
                    )

                # data = gr.Textbox(visible=lambda mode: mode == 'YouTube')
                with gr.Column(scale=1):
                    index_status = gr.Textbox("Not Index", label="Index Status")
                    upload_button.upload(
                        upload, inputs=upload_button, outputs=[url, index_status]
                    )

                    index_button = gr.Button("Start Indexing", variant="primary")
                    index_button.click(
                        indexing, inputs=[mode, url], outputs=[index_status]
                    )
                    stop_index_button = gr.Button("Stop Indexing")
                    stop_index_button.click(kill_index, outputs=[index_status])


def main():
    demo.launch(inbrowser=True, share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Ollama")
    parser.add_argument("--version", action="version", version="Chat with Ollama")
    main()
