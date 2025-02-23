from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import json
from nlp import chunking
def get_vectorstore(chunked_data):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunked_text = [d["text"] for d in chunked_data]
    metadata = [d["metadata"] for d in chunked_data]
    vectorstore = FAISS.from_texts(chunked_text, embeddings, metadatas=metadata)
    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain

def process_data():
    with open("data/processed/cleaned.json",encoding="utf-8") as f:
        data = json.load(f)
        chunked_data = chunking.chunk_data(data)
        vectorstore = get_vectorstore(chunked_data)
        chain = get_conversation_chain(vectorstore)
        return chain
"""
load_dotenv()

with open("../data/processed/cleaned.json",encoding="utf-8") as f:
    data = json.load(f)
    chunked_data = chunking.chunk_data(data)
    vectorstore = get_vectorstore(chunked_data)
    chain = get_conversation_chain(vectorstore)
    query = "parle moi de la filére info??"
    response = chain.invoke({"question": query})

    # Print the response
    print(response["answer"])


"""