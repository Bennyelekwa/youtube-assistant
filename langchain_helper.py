"""LangChain glue for the YouTube Assistant.

Two pieces:
- `create_db_from_youtube_video_url`: download a transcript, chunk it, embed
  it with OpenAI, and stash it in a FAISS vector store.
- `stream_response_from_query`: retrieve the most relevant chunks for a question
  and stream a grounded answer back, also returning the retrieved chunks so the
  UI can render them as sources.

The OpenAI API key is passed in by the caller so the same code path serves both
the server-held key and the user-pasted "bring your own key" flow.
"""

from __future__ import annotations

from typing import Generator

from langchain_community.document_loaders import youtube
from langchain_community.vectorstores import faiss
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AuthenticationError, RateLimitError


_PROMPT = PromptTemplate(
    input_variables=["question", "docs"],
    template=(
        "You are a helpful assistant that can answer questions about YouTube "
        "videos based on the video's transcript.\n\n"
        "Answer the following question: {question}\n"
        "By searching the following video transcript: {docs}\n\n"
        "Only use the factual information from the transcript to answer the "
        "question. If you feel like you don't have enough information to "
        "answer the question, say \"I don't know\". Your answers should be "
        "verbose and detailed."
    ),
)


class ServerKeyUnusable(Exception):
    """Raised when the server-held OpenAI key is out of credit, revoked, or
    otherwise unusable. The UI catches this to switch into the BYOK flow."""


def is_server_key_unusable(err: Exception) -> bool:
    """Return True only for errors that mean 'this key will never work as-is':
    invalid/revoked credentials, or `insufficient_quota`. Transient 429s
    (rate limits other than quota exhaustion) should be retried, not BYOK-ed."""
    if isinstance(err, AuthenticationError):
        return True
    if isinstance(err, RateLimitError):
        code = getattr(err, "code", None)
        if not code:
            body = getattr(err, "body", None) or {}
            code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
        return code == "insufficient_quota"
    return False


def create_db_from_youtube_video_url(video_url: str, openai_api_key: str) -> faiss.FAISS:
    """Fetch the transcript for `video_url` and build an in-memory FAISS store
    over its 1000-character chunks."""
    loader = youtube.YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return faiss.FAISS.from_documents(docs, embeddings)


def stream_response_from_query(
    db: faiss.FAISS,
    query: str,
    openai_api_key: str,
    k: int = 4,
) -> tuple[Generator[str, None, None], list[Document]]:
    """Retrieve the top-k transcript chunks for `query` and return:
    1. A generator yielding the answer token-by-token (consumed by `st.write_stream`).
    2. The list of retrieved `Document` objects so the UI can show grounded sources.

    gpt-3.5-turbo-instruct has a 4097-token context window; chunk_size=1000 with
    k=4 leaves comfortable headroom for the prompt and the streamed response.
    """
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join(d.page_content for d in docs)

    llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)
    chain = _PROMPT | llm | StrOutputParser()

    token_stream = chain.stream({"question": query, "docs": docs_page_content})
    return token_stream, docs
