import time

import streamlit as st
import os
from dotenv import load_dotenv
import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
def init():
    # Load environment variables
    load_dotenv()
    api_key=os.getenv("GROQ_API_KEY")
    if api_key is None:
        logging.error("GROQ_API_KEY not found in .env file")
        st.error("GROQ_API_KEY not found in .env file")
        st.stop()

    # Setup Streamlit page
    st.set_page_config(
        page_title="INSA Chatbot",
        page_icon="🎒"
    )
    st.header("Welcome to INSA Chatbot")

def create_streamlit_UI(title, description):
    st.title(title)
    st.markdown(description)

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



if __name__ == '__main__':
    main()