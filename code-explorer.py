from pathlib import Path

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language

repo_path = Path.home() / "code-reference" / "spring-petclinic-microservices"

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".java"],
    parser=LanguageParser(language=Language.JAVA, parser_threshold=500),
)

documents = loader.load()
print(len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=2000, chunk_overlap=200
)

texts = code_splitter.split_documents(documents)
print(len(texts))

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

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

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="mistral:latest")
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
question = "What are the different API methods?"
result = qa(question)
print(result)
