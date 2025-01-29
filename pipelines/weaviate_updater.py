import weaviate
from langchain_community.vectorstores import Weaviate

client = weaviate.Client(url="http://localhost:8080")
def update_vector_db(texts, embeddings):
    Weaviate.from_texts(
        texts=texts,
        embedding=embeddings,
        client=client,
        class_name="INSAData"
    )