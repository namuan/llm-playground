#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "langchain",
#   "langchain-community",
#   "langchain-ollama",
#   "langchain-chroma",
# ]
# ///
from pathlib import Path

from langchain import hub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import (
    LanguageParser,
)
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

repo_path = Path.home() / "code-reference" / "spring-petclinic-microservices"

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".java"],
    parser=LanguageParser(language=Language.JAVA, parser_threshold=500),
)

documents = loader.load()
print(f"Total documents found: {len(documents)}")

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=2000, chunk_overlap=200
)

texts = code_splitter.split_documents(documents)

DB_DIRECTORY = Path.cwd() / "target"
embedding_provider = OllamaEmbeddings(model="nomic-embed-text:latest")
if DB_DIRECTORY.exists():
    db = Chroma(embedding_function=embedding_provider, persist_directory="target")
else:
    DB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(
        texts,
        embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
        persist_directory="target",
    )
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

llm = ChatOllama(model="llama3.2:latest")
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
question = "What are the different API methods?"
result = qa(question)
print(result)
