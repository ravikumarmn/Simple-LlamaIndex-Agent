import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import json
from langchain.agents import Tool
from src.agent import VectorDBTool, get_agent, InappropriateContentDetector

# from dotenv import load_dotenv

# load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

qa_system_prompt = """
You are an expert educational system trusted for providing accurate explanations based on NCERT concepts.
Your task is to explain topics using only the provided context. If the context does not contain the answer, clearly state that the information is not available.
Your explanation should be presented as a well-structured, informative article aimed at helping students and educators understand the subject matter deeply.

Your response should focus on:
- A brief introduction to the concept, based on the NCERT material.
- A detailed explanation of the key principles or examples, using relevant information from the context.
- Any key observations, facts, or results that help explain the topic further.
- A concluding summary or key takeaways that reinforce the main points.

Ensure that the response is:
- Clear and precise, without restating the query.
- Focused on explaining the concept without introducing unrelated information.
- Organized for easy understanding by students and educators.

Use this tool to query the Vector Database for complex information. Do NOT use this for casual greetings or simple queries.

Query: {query_str}
"""


def get_service_config():
    from src.service_config import service_config

    return service_config


class QueryRequest(BaseModel):
    query: str = "what is sound propagation?"


def get_retrieval():
    from retrieval import SimpleRetrieverGeneration

    ncert_retriever_generation = SimpleRetrieverGeneration()
    return ncert_retriever_generation


@app.get("/")
async def home():
    return {"response": "Success!"}


# @app.post("/query")
# def query_rag(request: QueryRequest):
#     try:
#         query_str = request.query
#         ncert_retriever_generation = get_retrieval()
#         response, is_valid, _ = ncert_retriever_generation.get_query_response(query_str)
#         if not is_valid:
#             return {
#                 "response": "Query seems to be irrelevant. Kindly elaborate or rephrase your query."
#             }
#         return {"response": response[12:]}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent")
def query_with_agent(request: QueryRequest):
    try:
        query_str = request.query
        ncert_search = json.load(open("./config/ncert_search.json"))
        tools = [
            Tool(
                name="VectorDBTool",
                func=VectorDBTool().run,
                description="Use this tool to query the VectorDB for complex information.",
            ),
            Tool(
                name="InappropriateContentDetector",
                func=InappropriateContentDetector().run,
                description="Detects inappropriate language in user queries and warns the user.",
            ),
            # Tool(name="VulnerabilityTool", func=VulnerabilityTool().run,
            #      description="Dynamically detects vulnerability-related queries and responds with explanations and mitigation strategies."),
        ]

        model_name = ncert_search["agent_llm"]
        agent = get_agent(tools, model_name)
        # response = agent.invoke(qa_system_prompt.format(query_str=query_str))
        response = agent.invoke(query_str)
        return {"response": response["output"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
