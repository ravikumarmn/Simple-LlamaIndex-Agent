import json
from langchain.agents import initialize_agent, Tool, AgentType

# from langchain.chat_models.openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import List

# from langchain.s import LLMChain
from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())


def get_retrieval():
    from retrieval import SimpleRetrieverGeneration

    ncert_retriever_generation = SimpleRetrieverGeneration()
    return ncert_retriever_generation


# class GeneralLLMTool(BaseTool):
#     name: str = "GeneralLLMTool"
#     description: str = "Use this tool to handle greetings only."

#     def _run(self, query: str):
#         llm = get_agent_llm()
#         response = llm(query)
#         return response

# class VulnerabilityTool(BaseTool):
#     name: str = "VulnerabilityTool"
#     description: str = "This tool dynamically decides if a query is related to vulnerabilities and responds with an explanation and mitigation strategies."

#     def _run(self, query: str):
#         llm = get_agent_llm()

#         # Dynamic prompt for the LLM to determine if the query is vulnerability-related
#         vulnerability_prompt = f"""
#         The following query has been provided:
#         "{query}"

#         Please analyze the query and determine if it is related to a security vulnerability or software flaw.
#         If it is, provide an explanation of the vulnerability and suggest mitigation strategies.
#         If it is not, simply state that the query is not related to vulnerabilities.
#         """

#         response = llm(vulnerability_prompt)
#         return response


class InappropriateContentDetector(BaseTool):
    name: str = "InappropriateContentDetector"
    description: str = (
        "This tool detects inappropriate or offensive language in user queries and provides a warning if detected."
    )

    def _run(self, query: str):
        llm = get_agent_llm()
        content_moderation_prompt = f"""
        Analyze the following query and determine if it contains inappropriate, offensive, or harmful language:
        "{query}"

        If the language is inappropriate, generate a polite response asking the user to rephrase their query. 
        If the language is severely inappropriate, respond with a warning and do not engage further.
        """

        response = llm(content_moderation_prompt)
        return response


class VectorDBTool(BaseTool):
    name: str = "VectorDBTool"
    description: str = (
        "Use this tool to query the Vector Database for complex information. Do NOT use this for casual greetings or simple queries."
    )

    def _run(self, query: str):
        retriever = get_retrieval()
        response, is_valid, _ = retriever.get_query_response(query)
        if not is_valid:
            return "No relevant information found."
        return response


def get_agent_llm(model="gpt-4"):
    llm = ChatOpenAI(model=model, temperature=0.7)
    return llm


def get_agent(tools: List, model: str):
    agent = initialize_agent(
        tools=tools,
        llm=get_agent_llm(model),
        # Allows LLM to dynamically choose tools
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


if __name__ == "__main__":
    ncert_search = json.load(open("./config/ncert_search.json"))
    tools = [
        Tool(
            name="VectorDBTool",
            func=VectorDBTool().run,
            description="Use this tool to query the VectorDB for complex information.",
        ),
    ]
    model_name = ncert_search["agent_llm"]
    agent = get_agent(tools, model_name)

    response = agent.invoke("what is sound propogation?")
    print(response)
