"""
Ollama Playground
-----------------

A Streamlit app for playing with Ollama's models.

pip install -r requirements.txt
streamlit run ollama-playground.py
"""

import streamlit as st
from litellm import completion

from providers.ollama import OllamaProvider

st.set_page_config(page_title="Ollama Playground", page_icon="🦙")

ollama_provider = OllamaProvider()


def get_response(model, user_input, max_tokens, top_p):
    try:
        full_response = ""
        chat_completion = completion(
            model=f"ollama/{model}",
            messages=[{"role": "user", "content": user_input}],
            api_base="http://localhost:11434",
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )
        placeholder = st.empty()
        for message in chat_completion:
            full_response += message.choices[0].delta.content or ""
            placeholder.markdown(full_response + "▌")

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        return None
    except Exception as e:
        return str(e)


initial_message = {"role": "assistant", "content": "How may I assist you today?"}


# Clear chat history function and button
def clear_chat_history():
    st.session_state.messages = [initial_message]


def main():
    st.header("[Ollama](https://ollama.ai)")

    with st.sidebar:
        selected_model = st.sidebar.selectbox(
            "Choose a model:", ollama_provider.get_available_models()
        )
        max_tokens = st.slider(
            "Max Tokens",
            10,
            1000,
            500,
        )
        top_p = st.slider("Top P", 0.0, 1.0, 0.5, 0.05)

    if "messages" not in st.session_state:
        st.session_state.messages = [initial_message]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                error = get_response(selected_model, prompt, max_tokens, top_p)
                if error:
                    st.error(f"Error: {error}")

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
