# LangChain RAG Tutorial

This project is a small, notebook-first tutorial for building a Retrieval-Augmented Generation (RAG) workflow with LangChain, OpenAI embeddings, and ChromaDB.

The notebooks walk through two phases:

1. **Add documents:** load a text file, split it into chunks, embed each chunk, and persist the vectors to Chroma.
2. **Query documents:** load the persisted vector store, retrieve relevant chunks, send them to an LLM, and produce an answer grounded in the retrieved text.

The sample document is [`data/rag_sample1.txt`](data/rag_sample1.txt), a transcript from Security Now! episode 1073.

## Project Layout

```text
.
|-- data/
|   `-- rag_sample1.txt              # Source text used by the tutorial
|-- notebook/
|   |-- Rag1_Add_Docs.ipynb          # Phase 1: chunk, embed, and store
|   `-- Rag1_Query.ipynb             # Phase 2: retrieve and answer
|-- src/
|   |-- rag1_add_docs.py             # Script version of document ingestion
|   `-- rag1_query.py                # Script version of query flow
|-- pyproject.toml                   # Python dependencies
|-- justfile                         # Convenience commands
`-- README.md
```

## What You Will Build

The complete RAG flow looks like this:

```text
Document -> Chunks -> Embeddings -> Chroma vector store
Question -> Retriever -> Prompt + LLM -> Answer
```

The first notebook creates the vector database. The second notebook depends on that database already existing.

## Why Ingestion and Query Are Separate

In production RAG systems, document ingestion and user querying are usually separate workflows.

**Document ingestion** is a background or batch process. It reads source files, cleans or normalizes text, splits documents into chunks, creates embeddings, and writes those embeddings to a vector database. This work can be slow and expensive because every chunk must be sent through an embedding model. It should usually run only when documents are added, updated, or deleted.

**Querying** is the online user-facing process. It takes a user's question, searches the existing vector database, retrieves the most relevant chunks, and sends only those chunks to the LLM. This path needs to be fast, reliable, and available whenever users ask questions.

Keeping the two phases separate gives a production system several advantages:

- **Lower latency:** users do not wait for documents to be loaded, chunked, and embedded during every question.
- **Lower cost:** embeddings are created once and reused across many queries.
- **Independent scaling:** ingestion workers can run on a schedule, while query services can scale based on user traffic.
- **Better reliability:** a failed document update does not have to take down the query API.
- **Cleaner operations:** teams can monitor ingestion jobs, retry failed files, version vector stores, and deploy query changes independently.
- **Easier debugging:** retrieval problems can be isolated from document-processing problems.

In this tutorial, [`Rag1_Add_Docs.ipynb`](notebook/Rag1_Add_Docs.ipynb) represents the production ingestion job, and [`Rag1_Query.ipynb`](notebook/Rag1_Query.ipynb) represents the production query path.

## Prerequisites

- Python 3.12 or newer
- An OpenAI API key
- `uv` is recommended for environment setup

Install dependencies:

```bash
uv sync
```

If you prefer to use the `justfile`:

```bash
just setup
```

Create a local `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The project uses `python-dotenv`, so both notebooks and scripts call `load_dotenv()` to read this key.

## Tutorial Part 1: Add Documents

Open [`notebook/Rag1_Add_Docs.ipynb`](notebook/Rag1_Add_Docs.ipynb).

This notebook implements:

```text
Documents -> Chunk -> Embed -> Store
```

### 1. Choose the input file and vector store location

For local development, the notebook uses:

```python
file_location = "../data/rag_sample1.txt"
store_location = "../vector_db/chroma_db1"
```

`file_location` points to the source document. `store_location` is where Chroma will persist the generated vector database.

### 2. Load the environment

```python
from dotenv import load_dotenv

load_dotenv()
```

This loads `OPENAI_API_KEY` from `.env`.

### 3. Load the document

```python
from langchain_core.documents import Document

docs = [
    Document(page_content=open(file_location).read())
]
```

LangChain works with `Document` objects. Here, the full text file is wrapped as one document.

### 4. Split the document into chunks

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

splits = splitter.split_documents(docs)
```

RAG systems usually retrieve chunks, not whole documents. This tutorial uses:

- `chunk_size=500`: each chunk aims to be about 500 characters
- `chunk_overlap=100`: adjacent chunks share 100 characters to preserve context across boundaries

### 5. Embed and persist the chunks

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=store_location,
)
```

This sends each chunk to OpenAI's embedding model, stores the resulting vectors in Chroma, and persists the database under `vector_db/chroma_db1`.

After this notebook runs successfully, you should have a local vector store ready for querying.

## Tutorial Part 2: Query Documents

Open [`notebook/Rag1_Query.ipynb`](notebook/Rag1_Query.ipynb).

This notebook implements:

```text
Query -> Retrieve -> LLM -> Answer
```

Run the add-documents notebook first, because this notebook expects the vector database to already exist.

### 1. Point to the existing vector store

```python
store_location = "../vector_db/chroma_db1"
```

This must match the `store_location` used in the ingestion notebook.

### 2. Load the vector store and create a retriever

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory=store_location,
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

The retriever searches Chroma for the four chunks most relevant to the question.

### 3. Create the prompt and LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")
```

The prompt tells the model to answer only from retrieved context. `temperature=0` makes the response more deterministic.

### 4. Compose the RAG chain with LCEL

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

This chain:

1. Receives a question.
2. Sends the question to the retriever.
3. Formats retrieved chunks as context.
4. Inserts the context and question into the prompt.
5. Sends the prompt to the chat model.
6. Parses the model response as a string.

### 5. Ask a question

```python
response = rag_chain.invoke("What is this document about?")
print(response)
```

You can replace the question with anything related to the sample transcript, for example:

```python
rag_chain.invoke("What topics are covered in the episode?")
rag_chain.invoke("What does the transcript say about consumer routers?")
rag_chain.invoke("Who are the hosts?")
```

### 6. Debug retrieved chunks

The notebook includes a debug cell:

```python
docs = retriever.invoke("What is this document about?")

for i, d in enumerate(docs):
    print(f"chunk {i}: {d.page_content}")
    print("-" * 50)
```

Use this when an answer looks weak or incomplete. It shows which chunks were sent to the model as context.

## Running the Script Versions

The `src/` directory contains script versions of the notebook workflows.

Build the vector store:

```bash
uv run python src/rag1_add_docs.py
```

Query the vector store:

```bash
uv run python src/rag1_query.py
```

One small difference: the scripts currently use `vector_db/chroma_db1a`, while the notebooks use `vector_db/chroma_db1`. If you switch between notebooks and scripts, make sure both phases point to the same store location.

## Colab Notes

Both notebooks include commented cells for Google Colab:

- Mount Google Drive.
- Set document and Chroma paths under Drive.
- Load `OPENAI_API_KEY` from Colab user data.

For local development, keep the Colab cells commented and use the local path cells instead.

## Common Issues

### Missing API key

If you see an authentication error, confirm that `.env` exists in the project root and contains:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Vector store not found

Run [`notebook/Rag1_Add_Docs.ipynb`](notebook/Rag1_Add_Docs.ipynb) before [`notebook/Rag1_Query.ipynb`](notebook/Rag1_Query.ipynb).

Also confirm that both notebooks use the same `store_location`.

### Empty or irrelevant answers

Inspect the retrieved chunks with the debug cell. If the retrieved chunks do not contain the information needed to answer the question, try:

- asking a more specific question
- increasing `k` in `search_kwargs={"k": 4}`
- adjusting `chunk_size` and `chunk_overlap`
- adding more source documents

## Next Steps

Once the basic flow works, useful experiments include:

- Add metadata such as source filenames or page numbers to each `Document`.
- Load multiple files instead of one text file.
- Try different chunk sizes and overlaps.
- Change the embedding model.
- Add citations by returning retrieved chunk metadata with the final answer.
- Wrap the query chain in a small CLI or web app.
