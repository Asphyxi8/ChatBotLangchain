from fastapi import FastAPI
from agents.hospital_rag_agent import *
from models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput
from utils.async_utils import async_retry
app = FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot",
)

# @async_retry(max_retries=10, delay=1)
def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
  
    return hospital_rag_agent_executor.ainvoke(input={"input": str(query)})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/hospital-rag-agent")
def query_hospital_agent(query: HospitalQueryInput) -> HospitalQueryOutput:
    print(query)
    print(query.text)
    query_response = invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response