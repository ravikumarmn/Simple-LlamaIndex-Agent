import json
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader
from service_config import ServiceConfig
from datetime import datetime


class DocumentManager:
    """
    A class representing a collection of documents.
    """

    def __init__(self):
        service_config = ServiceConfig()
        self.index = service_config.system_indexer
        self.ncert_search = json.load(open("./config/ncert_search.json"))

    def prepare_pdf(self, files):
        documents = SimpleDirectoryReader(input_files=files).load_data()
        return documents

    def load_nodes(self, files):
        documents = self.prepare_pdf(files)
        # Parse the documents into chunks
        parser = SimpleNodeParser.from_defaults(
            chunk_size=self.ncert_search["chunk_size"],
            chunk_overlap=self.ncert_search["chunk_overlap"],
        )
        nodes = parser.get_nodes_from_documents(documents)

        # Add proper metadata to each node
        current_time = datetime.now().isoformat()
        for node in nodes:
            if hasattr(node, "metadata"):
                node.metadata["last_accessed_date"] = current_time

        return nodes

    def index_doc_from_files(self, files):
        nodes = self.load_nodes(files)
        self.index.insert_nodes(nodes)
        return len(nodes)


document_manager = DocumentManager()
