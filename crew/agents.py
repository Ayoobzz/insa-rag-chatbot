import os
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from crew.tools import rag_search, timetable_lookup

def setup_agents():
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

    rag_tool = Tool(
        name="RAG_Search",
        func=rag_search,
        description="Search the INSA Rennes knowledge base using RAG."
    )
    timetable_tool = Tool(
        name="Timetable_Lookup",
        func=timetable_lookup,
        description="Fetch timetable information for INSA Rennes students."
    )

    main_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the central assistant for INSA Rennes students. Use the conversation history to provide context-aware responses. Answer directly if the query is simple (e.g., about the date or basic info). Otherwise, decide whether to use the RAG_Search tool for general queries or the Timetable_Lookup tool for timetable-related queries. If unsure, use RAG_Search."),
    ])

    timetable_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in INSA Rennes schedules and timetables. Use the conversation history to provide context-aware timetable information. Use the Timetable_Lookup tool to fetch accurate timetable details."),
    ])

    main_agent_llm = llm.bind_tools([rag_tool, timetable_tool])
    timetable_agent_llm = llm.bind_tools([timetable_tool])

    return {
        "Main_agent": {
            "llm": main_agent_llm,
            "prompt": main_prompt,
            "tools": [rag_tool, timetable_tool]
        },
        "Timetable_agent": {
            "llm": timetable_agent_llm,
            "prompt": timetable_prompt,
            "tools": [timetable_tool]
        }
    }