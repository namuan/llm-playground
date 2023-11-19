import openai

client = openai.OpenAI(base_url="http://0.0.0.0:8000")

import streamlit as st

st.set_page_config(page_title="CodeLlama Playground - via Ollama", page_icon="🦙")

st.image("https://images.emojiterra.com/twitter/v14.0/1024px/1f999.png", width=90)

API_KEY = "foo"

MODEL_CODELLAMA = "ollama/phind-codellama:latest"

print(client.models.list())

def get_response(api_key, model, user_input, max_tokens, top_p):
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_input}],
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return chat_completion.choices[0].message.content, None
    except Exception as e:
        raise e
        # return None, str(e)


st.header("Meta's `CodeLlama` via [Ollama](https://ollama.ai)")

with st.expander("About this app"):
    st.write(
        """
    Try the latest [CodeLlama model](https://about.fb.com/news/2023/08/code-llama-ai-for-coding/) from Meta with this Streamlit app. 

    🔧 For optimal responses, consider adjusting the `Max Tokens` value from `100` to `1000`.
    """
    )

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    max_tokens = st.slider(
        "Max Tokens",
        10,
        1000,
        500,
    )
    top_p = st.slider("Top P", 0.0, 1.0, 0.5, 0.05)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, error = get_response(
                st.session_state.api_key, MODEL_CODELLAMA, prompt, max_tokens, top_p
            )
            if error:
                st.error(f"Error: {error}")
            else:
                placeholder = st.empty()
                placeholder.markdown(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)


# Clear chat history function and button
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
