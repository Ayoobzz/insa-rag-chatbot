import streamlit as st
import os
from dotenv import load_dotenv
import logging
from rich.console import Console
from rich.logging import RichHandler
from crew.agents import setup_agents
from crew.tasks import execute_task

# Setup logging with Rich
console = Console()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


def init():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("GROQ_API_KEY not found in .env file")
        st.error("GROQ_API_KEY not found in .env file")
        st.stop()


def create_streamlit_UI(title, description):
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "text": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        try:
            response = execute_task(user_input, agents, st.session_state.messages)
            logging.debug(f"Processed response: {response}")
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}", exc_info=True)
            response = f"An error occurred: {str(e)}. Please try again."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "text": response})


def main():
    init()
    create_streamlit_UI("INSA Chatbot", "Ask me anything about INSA!")
    if "agents" not in st.session_state:
        st.session_state.agents = setup_agents()
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
    user_input = st.chat_input("Ask me anything about INSA Rennes", key="user_input")
    if user_input:
        handle_input(user_input, st.session_state.agents)


if __name__ == "__main__":
    main()