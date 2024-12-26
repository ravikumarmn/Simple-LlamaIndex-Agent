import json
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# Local Imports
import prompts
from llms import get_openai_model
from indexer import load_pinecone_index, get_rag_index


def get_model_by_name(name, system_prompt):
    if name.startswith("gpt"):
        return get_openai_model(name, system_prompt)
    else:
        raise NotImplementedError(f"Model {name} not implemented")


class ServiceConfig:

    ncert_search = json.load(open("./config/ncert_search.json"))
    Settings.llm = get_openai_model(ncert_search["llm"])
    Settings.embed_model = OpenAIEmbedding(
        model=ncert_search["embedding_model_name"],
        dimensions=ncert_search["embedding_dim"],
    )
    system_indexer = get_rag_index(
        load_pinecone_index(
            ncert_search["index_name"], host=ncert_search["pinecone_host"]
        ),
        namespace=ncert_search["namespace"],
    )

    MODEL = get_model_by_name(
        ncert_search["llm"], system_prompt=prompts.qa_system_prompt
    )


service_config = ServiceConfig()
