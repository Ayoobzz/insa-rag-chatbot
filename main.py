import time
import streamlit as st
import os
from dotenv import load_dotenv
import logging
from rich.console import Console
from rich.logging import RichHandler
import vectorstore.utils as utils

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
def init():
    load_dotenv()
    api_key=os.getenv("GROQ_API_KEY")
    if api_key is None:
        logging.error("GROQ_API_KEY not found in .env file")
        st.error("GROQ_API_KEY not found in .env file")
        st.stop()

    st.set_page_config(
        page_title="INSA Chatbot",
        page_icon="🎒"
    )
    st.header("Welcome to INSA Chatbot")

def create_streamlit_UI(title, description):
    st.title(title)
    st.markdown(description)

    with st.sidebar:
        st.subheader("Your PDFs 📖")
        st.text_input("Enter a question")


def handle_input(user_input):
    st.session_state.messages.append({"role": "user", "text": user_input})
    if st.session_state.chain:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            time.sleep(2)
            input_data = {
                'question': user_input,
                'chat_history': st.session_state.messages
            }

            response = st.session_state.chain(input_data)
        answer = response.get('answer', 'No answer provided.')
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "text": answer})
    else:
        st.error("⚠️ Please upload and process a PDF first.")

def main():
    init()
    create_streamlit_UI("INSA Chatbot", "Ask me anything about INSA!")
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.chain= utils.process_data()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    user_input = st.chat_input("Ask me anything about INSA Rennes", key="user_input")
    if user_input:
        handle_input(user_input)



if __name__ == '__main__':
    main()