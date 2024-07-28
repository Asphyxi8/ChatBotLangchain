import os
import requests
import streamlit as st
import os

# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyD4qN5YnU7q6NWApe62o4cwT-7rs82UR_4'
os.environ['NEO4J_URI'] = 'neo4j+s://54113544.databases.neo4j.io'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'uLebAk4boUnAEUlUs_Rf5lCXjwrgP-_cL4M5GYsonLk'

os.environ['HOSPITALS_CSV_PATH'] = 'https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/hospitals.csv'
os.environ['PAYERS_CSV_PATH'] = 'https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/payers.csv'
os.environ['PHYSICIANS_CSV_PATH'] = 'https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/physicians.csv'
os.environ['PATIENTS_CSV_PATH'] = 'https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/patients.csv'
os.environ['VISITS_CSV_PATH'] = 'https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/visits.csv'
os.environ['REVIEWS_CSV_PATH'] = 'https://raw.githubusercontent.com/hfhoffman1144/langchain_neo4j_rag_app/main/data/reviews.csv'

os.environ['HOSPITAL_AGENT_MODEL'] = 'gemini-1.5-flash'
os.environ['HOSPITAL_CYPHER_MODEL'] = 'gemini-1.5-flash'
os.environ['HOSPITAL_QA_MODEL'] = 'gemini-1.5-flash'

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from chains.hospital_review_chain import reviews_vector_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)


HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")


# ...

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]


# ...

chat_model = ChatGoogleGenerativeAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)

hospital_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
response = hospital_rag_agent_executor.invoke(
    {"input": "What is the wait time at Wallace-Hamilton?"}
)

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/hospital-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the hospitals, patients,
        visits, physicians, and insurance payers in  a fake hospital system.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("- Which hospitals are in the hospital system?")
    st.markdown("- What is the current wait time at wallace-hamilton hospital?")
    st.markdown(
        "- At which hospitals are patients complaining about billing and "
        "insurance issues?"
    )
    st.markdown("- What is the average duration in days for closed emergency visits?")
    st.markdown(
        "- What are patients saying about the nursing staff at "
        "Castaneda-Hardy?"
    )
    st.markdown("- What was the total billing amount charged to each payer for 2023?")
    st.markdown("- What is the average billing amount for medicaid visits?")
    st.markdown("- Which physician has the lowest average visit duration in days?")
    st.markdown("- How much was billed for patient 789's stay?")
    st.markdown(
        "- Which state had the largest percent increase in medicaid visits "
        "from 2022 to 2023?"
    )
    st.markdown("- What is the average billing amount per day for Aetna patients?")
    st.markdown("- How many reviews have been written from patients in Florida?")
    st.markdown(
        "- For visits that are not missing chief complaints, "
        "what percentage have reviews?"
    )
    st.markdown(
        "- What is the percentage of visits that have reviews for each hospital?"
    )
    st.markdown(
        "- Which physician has received the most reviews for this visits "
        "they've attended?"
    )
    st.markdown("- What is the ID for physician James Cooper?")
    st.markdown(
        "- List every review for visits treated by physician 270. Don't leave any out."
    )

st.title("Hospital System Chatbot")
st.info(
    "Ask me questions about patients, visits, insurance payers, hospitals, "
    "physicians, reviews, and wait times!"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])
            
        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    # data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        # response = requests.post(CHATBOT_URL, json=data)
        # response = hospital_rag_agent_executor.invoke(
        #     {"input": "What is the wait time at Wallace-Hamilton?"}
        # )
        # print(response)
    
        st.warning(prompt)
        response = hospital_rag_agent_executor.invoke({"input": prompt})
        st.warning(response)
        if response:
            output_text = response["output"]
            explanation = response["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )