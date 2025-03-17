import time
import streamlit as st
import os
from dotenv import load_dotenv
import logging
from rich.console import Console
from rich.logging import RichHandler
from crewai import Crew
from crew.agents import setup_agents
from crew.tasks import create_rag_task

# Setup logging with Rich
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

def init():
    """Initialize environment variables and check for API keys."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("GROQ_API_KEY not found in .env file")
        st.error("GROQ_API_KEY not found in .env file")
        st.stop()

def create_streamlit_UI(title, description):
    """Set up the Streamlit UI with title, description, and sidebar."""
    st.set_page_config(page_title="INSA Chatbot", page_icon="🎒")
    st.header("Welcome to INSA Chatbot")
    st.title(title)
    st.markdown(description)
    with st.sidebar:
        st.image("assets/chatbot.png")
        st.title("Settings")
        st.subheader("Your API Keys 🗝️")
        st.text_input("Jina API Key", type="password")
        st.text_input("GROQ API Key", type="password")

def handle_input(user_input, agents):
    """Process user input using CrewAI agents and display the response."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        # Create a task for the main agent
        task = create_rag_task(agents["Main_agent"], user_input)

        # Set up and run the crew
        crew = Crew(
            agents=[agents["Main_agent"], agents["Timetable_agent"]],
            tasks=[task],
            verbose=True
        )
        time.sleep(2)
        response = crew.kickoff()

        # Display and store the response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "text": response})

def main():
    """Main function to run the Streamlit app."""
    init()
    create_streamlit_UI("INSA Chatbot", "Ask me anything about INSA!")

    # Initialize session state for messages and agents
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agents" not in st.session_state:
        st.session_state.agents = setup_agents(llm=None)  # Use default Groq LLM

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    # Handle user input
    user_input = st.chat_input("Ask me anything about INSA Rennes", key="user_input")
    if user_input:
        handle_input(user_input, st.session_state.agents)

if __name__ == "__main__":
    main()