import os
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import pickle

from src.global_settings import INDEX_STORAGE, CACHE_FILE


def build_indexes():
    """
    Build or load vector index using LangChain and ChromaDB from cached JSON data.
    Automatically persists the index to INDEX_STORAGE.

    Returns:
        Chroma vectorstore object
    """
    # Create INDEX_STORAGE directory if it doesn't exist
    os.makedirs(INDEX_STORAGE, exist_ok=True)

    # Check if cache file exists
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache file not found at {CACHE_FILE}")

    # Read data from cache file
    with open(CACHE_FILE, "rb") as f:
        try:
            cached_data = pickle.load(f)
            print("Cache file found. Running using cache...")
        except Exception as e:
            raise ValueError(f"Error reading cache file: {e}")

    # Convert cached data to list of Documents
    if isinstance(cached_data[0], Document):
        documents = cached_data
    else:
        documents = [
            Document(
                page_content=doc["page_content"],
                metadata=doc.get("metadata", {})
            )
            for doc in cached_data
        ]

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=INDEX_STORAGE)

    # Get or create collection
    collection = client.get_or_create_collection(
        name="vector",
        embedding_function=None,  # LangChain will handle embeddings
        metadata={"hnsw:space": "cosine"}
    )

    # Check if collection is empty before adding documents
    if collection.count() == 0:
        # Generate embeddings for documents
        document_embeddings = embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )

        # Prepare simplified metadata
        simplified_metadatas = []
        for doc in documents:
            metadata = doc.metadata.copy()
            # Remove or simplify complex fields
            if 'input_documents' in metadata:
                del metadata['input_documents']  # Remove problematic field
            # Ensure all metadata values are simple types
            simplified_metadata = {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            simplified_metadatas.append(simplified_metadata)

        # Add documents to collection
        collection.add(
            ids=[str(i) for i in range(len(documents))],
            documents=[doc.page_content for doc in documents],
            embeddings=document_embeddings,
            metadatas=simplified_metadatas
        )
        print(f"Added {len(documents)} documents to vector store")
    else:
        print("Using existing vector store")

    print(f"Number of documents in vector store: {collection.count()}")
    return collection
