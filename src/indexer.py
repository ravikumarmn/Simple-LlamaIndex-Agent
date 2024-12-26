import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv, find_dotenv
import json
from functools import lru_cache
import json

# _ = load_dotenv(find_dotenv())

# TOOL: pinecone_index.describe_index_stats()
# TOOL: pinecone_index.delete(delete_all=True, namespace='')


def get_document_manager():
    from document_manager import document_manager

    return document_manager


@lru_cache
def load_pinecone_index(index_name, host=""):
    """Base Index from pinecone
    This can be used to create multi-tenant using llama-index vector-store object"""
    ncert_search = json.load(open("./config/ncert_search.json"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if host == "":
        if index_name not in pc.list_indexes().names():
            # create the index
            print(f"Creating Index {index_name}")
            pc.create_index(
                name=index_name,
                dimension=ncert_search["embedding_dim"],
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        print(f"About {index_name} Index :\n{pc.describe_index(index_name)}")
    pinecone_index = pc.Index(
        index_name, host=host
    )  
    return pinecone_index


def get_rag_index(pinecone_index, namespace=""):
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace=namespace
    )
    # print("INDEX NAME :", vector_store.index_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store  # , insert_batch_size=20
    )
    return index


def delete(pc, index_name):
    try:
        pc.delete_index(index_name)
        print(f"{index_name} is deleted!")
    except Exception as e:
        print("Error :", e)


# Add this to indexer.py's main block
if __name__ == "__main__":
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # index = pc.Index("ncert-index")
    # stats = index.describe_index_stats()
    # print(f"Index stats: {stats}")
    # 1. Delete existing index

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if "ncert-index" in pc.list_indexes().names():
        pc.delete_index("ncert-index")

    # 2. Create new index with correct dimensions
    pc.create_index(
        name="ncert-index",
        dimension=3072,  # Updated for text-embedding-3-large
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    index = pc.Index("ncert-index")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")

    # 3. Index your documents
    files = ["data/iesc111.pdf"]
    doc_manager = get_document_manager()
    n_indexed = doc_manager.index_doc_from_files(files)
    print(f"Indexed {n_indexed} chunks")

# if __name__ == "__main__":
#     files = ["data/iesc111.pdf"]
#     doc_manager = get_document_manager()  # Delay the import
#     try:
#         n_indexed = doc_manager.index_doc_from_files(files)
#     except Exception as e:
#         print("An error occurred while indexing the documents.")
#         print(f"Error details: {e}")
#     print(f"Number of Indexed Nodes: {n_indexed}")

#     # # pinecone_index.delete(delete_all=True, namespace="ncert")
#     # pinecone_index = pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     # delete(pinecone_index, "ncert-index")
