import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


STORE_LOCATION = Path("vector_db/chroma_db1a")
RETRIEVAL_K = 4
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_QUESTION = "What is this document about?"


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_retriever(store_location: Path):
    logging.info("Loading vector store from %s", store_location)
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=str(store_location),
        embedding_function=embedding,
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def build_rag_chain(retriever):
    logging.info("Building RAG chain with model %s", CHAT_MODEL)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
"""
    )
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    load_dotenv()
    logging.info("Environment variables loaded")

    if not STORE_LOCATION.exists():
        raise FileNotFoundError(f"Vector store not found: {STORE_LOCATION}")

    question = DEFAULT_QUESTION
    retriever = build_retriever(STORE_LOCATION)
    rag_chain = build_rag_chain(retriever)

    logging.info("Running query: %s", question)
    response = rag_chain.invoke(question)
    print(response)

    docs = retriever.invoke(question)
    logging.info("Retrieved %s chunk(s)", len(docs))
    for index, doc in enumerate(docs, start=1):
        print(f"\nChunk {index}")
        print(doc.page_content)


if __name__ == "__main__":
    main()
