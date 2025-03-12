from crewai import Task
from tools import rag_search, timetable_lookup  # Import tools explicitly

def create_rag_task(agent, query: str) -> Task:
    """Create a task for answering a general query using RAG."""
    return Task(
        description=f"Answer the student's query: '{query}' using the RAG search tool.",
        agent=agent,
        expected_output="A detailed and accurate response to the student's query.",
        tools=[rag_search]
    )

def create_timetable_task(agent, query: str) -> Task:
    """Create a task for fetching timetable information."""
    return Task(
        description=f"Fetch timetable information for the query: '{query}'.",
        agent=agent,
        expected_output="Timetable details relevant to the query.",
        tools=[timetable_lookup]
    )