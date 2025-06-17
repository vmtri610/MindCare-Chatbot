from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
import os
import pickle
from typing import List
from dotenv import load_dotenv
from src.global_settings import FILES_PATH, CACHE_FILE
from src.prompts import CUSTOM_SUMMARY_EXTRACT_TEMPLATE

load_dotenv()

# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def ingest_documents() -> List[Document]:
    # Load documents
    documents = []
    for file_path in FILES_PATH:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        if file_path.endswith(".pdf"):  # Assuming PDF files, adjust as needed
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Set document ID as filename
            for doc in docs:
                doc.metadata["id"] = os.path.basename(file_path)
                documents.append(doc)
        else:
            # For text files or other formats
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"id": os.path.basename(file_path)}
                )
                documents.append(doc)

    for doc in documents:
        print(doc.metadata["id"])

    # Check for cache
    try:
        with open(CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
            print("Cache file found. Running using cache...")
            return cached_data
    except FileNotFoundError:
        print("No cache file found. Running without cache...")

    # Initialize components
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=512
    )
    text_splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=20
    )
    embeddings = OpenAIEmbeddings()

    # Process documents
    processed_docs = []

    for doc in documents:
        # Split document into chunks
        chunks = text_splitter.split_text(doc.page_content)

        # Summarize each document
        summary_chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=CUSTOM_SUMMARY_EXTRACT_TEMPLATE
        )
        summary = summary_chain.invoke(
            {"input_documents": [Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks]}
        )

        # Create new Document objects for each chunk with summary in metadata
        for chunk in chunks:
            embedded_chunk = embeddings.embed_query(chunk)
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    "id": doc.metadata["id"],
                    "summary": summary,
                    "embedding": embedded_chunk
                }
            )
            processed_docs.append(chunk_doc)

    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(processed_docs, f)
        print(f"Cache saved to {CACHE_FILE}")

    return processed_docs
