from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


FILE_LOCATION = Path("data/rag_sample1.txt")
STORE_LOCATION = Path("vector_db/chroma_db1")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(file_location: Path) -> list[Document]:
    text = file_location.read_text(encoding="utf-8")
    return [Document(page_content=text)]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(splits: list[Document], store_location: Path) -> Chroma:
    store_location.mkdir(parents=True, exist_ok=True)

    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(store_location),
    )


def main() -> None:
    load_dotenv()

    if not FILE_LOCATION.exists():
        raise FileNotFoundError(f"Document file not found: {FILE_LOCATION}")

    documents = load_documents(FILE_LOCATION)
    splits = split_documents(documents)
    build_vector_store(splits, STORE_LOCATION)

    print(f"Loaded {len(documents)} document(s) from {FILE_LOCATION}")
    print(f"Created {len(splits)} chunk(s)")
    print(f"Persisted Chroma vector store to {STORE_LOCATION}")


if __name__ == "__main__":
    main()
