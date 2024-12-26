from prompts import QA_PROMPT_TMPL, query_clf_prompt
from llama_index.core.schema import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import Optional, List
from llama_index.core.schema import NodeWithScore

from llama_index.core import get_response_synthesizer
from service_config import service_config

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.settings import Settings

# _ = load_dotenv(find_dotenv())


class LLMIncludeALLFieldsPostprocessor(BaseNodePostprocessor):
    exclude_keys_to_allow_all: list[str] = []

    @classmethod
    def class_name(cls) -> str:
        return "CaseSortPostprocessor"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        for node_with_score in nodes:
            node_with_score.node.excluded_llm_metadata_keys = (
                self.exclude_keys_to_allow_all
            )
        return nodes


SIMILARITY_TOP_K = service_config.ncert_search["similarity_top_k"]
LONG_ANSW_TOP_K = service_config.ncert_search["long_answ_top_k"]
llm_field_postprocessor = LLMIncludeALLFieldsPostprocessor(
    exclude_keys_to_allow_all=service_config.ncert_search["exclude_keys_to_allow_all"]
)

post_processors = [llm_field_postprocessor]


EMPTY_RESPONSE_REPLY_STR = """Your query did not receive a response from our server.

This might occur if there is no relevant information for your query or if the query is too vague. Please try again by framing your query as a direct question."""


class SimpleRetrieverGeneration:
    def __init__(self, response_mode="compact_accumulate") -> None:
        SIMILARITY_TOP_K = service_config.ncert_search["similarity_top_k"]
        self.post_processors = post_processors
        self.simple_retriever = VectorIndexRetriever(
            index=service_config.system_indexer, similarity_top_k=SIMILARITY_TOP_K
        )
        self.query_classification_model_name = service_config.ncert_search[
            "query_classification_model_name"
        ]
        self.response_mode = response_mode
        self.query_engine = self.create_query_engine()

    def create_query_engine(self):
        prompt_helper = None
        Settings.llm = service_config.MODEL
        response_synthesizer = get_response_synthesizer(
            response_mode=self.response_mode,
            use_async=False,
            streaming=False,
            text_qa_template=QA_PROMPT_TMPL,
            prompt_helper=prompt_helper,
        )
        simple_query_engine = RetrieverQueryEngine(
            retriever=self.simple_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self.post_processors,
        )
        return simple_query_engine

    def complete_query(self, query_str):
        response = service_config.MODEL.complete(
            query_clf_prompt.format(query_str=query_str)
        )
        return response.text

    def get_query_response(self, query_str):
        return self._get_query_response(query_str)

    def _get_query_response(self, query_str):
        # classify query
        extra_info = {"query_str": query_str}
        clf_model_name = self.query_classification_model_name
        # response = self.complete_query(query_str)
        # print("Exception is added to get classification as True")
        response = "Yes"
        extra_info["query_clf_model_name"] = clf_model_name
        extra_info["query_class"] = response
        # if response.lower() == "no":
        #     return (
        #         f"Query seems to be irrelevent to the legal issue or case. Kindly elaborate or rephrase your query.",
        #         False,
        #         extra_info,
        #     )

        response = self.query_engine.query(query_str)
        extra_info["sources"] = [
            {"file_name": x.node.metadata["file_name"], "score": x.score}
            for x in response.source_nodes
        ]
        response = response.response
        extra_info["raw_response"] = response
        # extra_info["response_generation_model"] = model
        if response == "Empty Response":
            return (
                EMPTY_RESPONSE_REPLY_STR,
                False,
                extra_info,
            )
        else:
            return response, True, extra_info

    def get_retrieve_nodes(self, query_str):
        nodes = self.simple_retriever.retrieve(QueryBundle(query_str))
        return nodes


class NcertRetrieverGeneration(SimpleRetrieverGeneration):
    def __init__(self, response_mode="compact_accumulate"):
        SIMILARITY_TOP_K = service_config.ncert_search["similarity_top_k"]
        self.post_processors = post_processors
        self.task_type = service_config.ncert_search["task_type"]
        self.simple_retriever = VectorIndexRetriever(
            index=service_config.system_indexer, similarity_top_k=SIMILARITY_TOP_K
        )
        self.query_classification_model_name = service_config.ncert_search[
            "query_classification_model_name"
        ]
        self.response_mode = response_mode
        self.query_engine = self.create_query_engine()

    def get_query_response(self, query_str):
        response, is_valid, extra_info = self._get_query_response(query_str)
        return response, is_valid, extra_info


ncert_retriever_generation = NcertRetrieverGeneration()


if __name__ == "__main__":
    # Test query
    query_str = "What is sound propagation?"
    response = ncert_retriever_generation.get_query_response(query_str)
    print(f"Query: {query_str}")
    print(f"Response: {response}")
