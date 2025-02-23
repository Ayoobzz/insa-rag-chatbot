from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

def get_vectorstore(chunked_data):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunked_text = [d["text"] for d in chunked_data]
    metadata = [d["metadata"] for d in chunked_data]
    vectorstore = FAISS.from_texts(chunked_text,embeddings,metadata)
    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0
    )


    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain
