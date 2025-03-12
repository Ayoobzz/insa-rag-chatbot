import os
from crewai import Agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import rag_search
from tools import timetable_lookup



def setup_agents(llm) -> dict:
    if llm is None:
        load_dotenv()
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1000
        )
    main_agent = Agent(
        role="Main INSA Chatbot",
        goal="Assist INSA Rennes students by answering queries or delegating to specialists.",
        backstory="You're the central assistant for INSA Rennes students, coordinating with other agents.",
        tools=[rag_search],
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # Timetable Agent
    timetable_agent = Agent(
        role="Timetable Specialist",
        goal="Provide accurate timetable information to INSA Rennes students.",
        backstory="You're an expert in INSA Rennes schedules and timetables.",
        tools=[timetable_lookup],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    return {"Main_agent": main_agent, "Timetable_agent": timetable_agent}