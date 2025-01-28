#!/usr/bin/env python3
"""
Playing with OpenAI tools and function to call database
https://cookbook.openai.com/examples/how_to_call_functions_for_knowledge_retrieval
"""

import ast
import os
from csv import writer
from pathlib import Path

import arxiv
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from scipy import spatial
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

load_dotenv()

GPT_MODEL = "gpt-3.5-turbo-1106"
EMBEDDING_MODEL = "text-embedding-ada-002"
PAPERS_DIRECTORY = Path.cwd() / "target" / "papers"
PAPERS_DIRECTORY.mkdir(exist_ok=True, parents=True)
PAPERS_DATA_PATH = PAPERS_DIRECTORY / "arxiv_library.csv"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception {e}")
        return e


def get_articles(query, library=PAPERS_DATA_PATH, top_k=5):
    """Retrieves top_k articles based on user query sorted by relevance"""
    search = arxiv.Search(
        query=query, max_results=top_k, sort_by=arxiv.SortCriterion.Relevance
    )
    result_list = []
    arxiv_search_results = list(search.results())
    print(f"Found {len(arxiv_search_results)} results for {query}")
    for result in arxiv_search_results:
        print(f"Processing {result.title}")
        result_dict = {
            "title": result.title,
            "summary": result.summary,
            "article_url": [x.href for x in result.links][0],
            "pdf_url": [x.href for x in result.links][1],
        }
        result_list.append(result_dict)

        response = embedding_request(text=result.title)
        data_embedding = response.data[0].embedding
        download_file_path = PAPERS_DIRECTORY.joinpath(result._get_default_filename())
        if not download_file_path.exists():
            result.download_pdf(PAPERS_DIRECTORY.as_posix())
        file_reference = [
            result.title,
            download_file_path,
            data_embedding,
        ]

        with open(library, "a") as csv_file:
            csv_writer = writer(csv_file)
            csv_writer.writerow(file_reference)
            csv_file.close()

    return result_list


def strings_ranked_by_relatedness(
    query,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n=100,
):
    """Returns a list of strings and relatedness"""
    embedded_input_query_response = embedding_request(query)
    embedded_input_query = embedded_input_query_response.data[0].embedding
    strings_and_relatedness = [
        (row["filepath"], relatedness_fn(embedded_input_query, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatedness.sort(key=lambda x: x[1], reverse=True)
    strings, _ = zip(*strings_and_relatedness)
    return strings[:top_n]


def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    page_number = 0
    pdf_text = ""
    for page in pdf_reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"

    return pdf_text


def create_chunks(text, n):
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(1.5 * n):
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1

        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))

        yield tokens[i:j]
        i = j


def extract_chunk(content, template_prompt):
    prompt = template_prompt + content
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content


def summarize_text(query):
    """
    - Reads the arxiv_library.csv file including the embedding
    - Finds the closest file to the user query
    - Scrapes the text from the pdf and chunk it
    - Get summmary for each chunk in parallel
    - Combine into a final summary and return to user
    """
    summary_prompt = """Summarize this text from an academic paper.
    Extract any key points with reasoning.\n\nContent:"""
    library_df = pd.read_csv(PAPERS_DATA_PATH).reset_index()
    library_df.columns = ["title", "filepath", "embedding"]
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    pdf_file_path = strings[0]
    print(f"Chunking text from {pdf_file_path}")
    pdf_text = read_pdf(pdf_file_path)
    pdf_text_chunks = create_chunks(pdf_text, 1500)
    print("Fetching summary for chunks")
    summary_results = ""
    text_chunks = [tokenizer.decode(chunk) for chunk in pdf_text_chunks]
    for chunk in text_chunks:
        summary_results += extract_chunk(chunk, summary_prompt)

    print("Generating final summary")
    final_summary_prompt = f"""
    Write a summary collated from this collection of key points extracted from an academic paper.
    The summary should highlight the core argument, conclusions and evidence, and answer the user query.
    User query: {query}
    The summary should be structured in bulleted lists following the heading Core Argument, Evidence and Conclusions.
    Key points:\n{summary_results}\nSummary:\n
    """
    final_summary_response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": final_summary_prompt},
        ],
        temperature=0,
    )
    return final_summary_response.choices[0].message.content


def main():
    df = pd.DataFrame(list())
    df.to_csv(PAPERS_DATA_PATH)
    query = "DPO Learning"
    get_articles(query)
    result_output = summarize_text(query)
    print(result_output)


if __name__ == "__main__":
    main()
