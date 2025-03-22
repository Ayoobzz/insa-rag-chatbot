import os
from crewai.tools import tool
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from groq import Groq


def index_init(data, index_name="insa-chatbot", batch_size=32):
    """
    Initialize and populate a Pinecone index with embedded data.

    Args:
        data: List of dictionaries containing text and metadata
        index_name: Name of the Pinecone index
        batch_size: Number of items to process in each batch
    """
    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    # Initialize Pinecone
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        raise Exception(f"Failed to initialize Pinecone: {str(e)}")

    # Check if index exists, create if it doesn't
    if index_name not in pinecone.list_indexes().names():
        try:
            pinecone.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created new index: {index_name}")
        except PineconeException as e:
            raise Exception(f"Failed to create index: {str(e)}")
    else:
        print(f"Index {index_name} already exists, connecting to it.")

    # Connect to the index
    index = pinecone.Index(index_name)

    # Load the embedding model
    model_name = "jinaai/jina-embeddings-v3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise Exception(f"Failed to load model/tokenizer: {str(e)}")

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Prepare and upsert data in batches
    to_upsert = []
    total_items = 0

    for datum in data:
        try:
            url = datum["metadata"]["url"]
            texts = datum["text"]
            title = datum["metadata"]["title"]
            chunk_number = datum["metadata"]["chunk_number"]

            # Tokenize and compute embeddings
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

            # Prepare vector for upsert
            vector_id = f"{url}_{chunk_number}"
            vector_data = {
                "id": vector_id,
                "values": embeddings.squeeze().tolist(),
                "metadata": {
                    "title": title,
                    "text": texts,
                    "url": url,
                    "chunk_number": chunk_number
                }
            }
            to_upsert.append(vector_data)

            # Upsert in batches
            if len(to_upsert) >= batch_size:
                index.upsert(vectors=to_upsert)
                total_items += len(to_upsert)
                print(f"Upserted batch of {len(to_upsert)} items")
                to_upsert = []

        except KeyError as e:
            print(f"Skipping item due to missing key: {str(e)}")
            continue
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue

    # Upsert any remaining items
    if to_upsert:
        index.upsert(vectors=to_upsert)
        total_items += len(to_upsert)

    print(f"Successfully indexed {total_items} items into {index_name}.")
    return index


def connect_to_index(index_name="insa-chatbot"):
    """
    Connect to an existing Pinecone index.

    Args:
        index_name: Name of the Pinecone index

    Returns:
        Pinecone Index object
    """
    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    # Initialize Pinecone
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        raise Exception(f"Failed to initialize Pinecone: {str(e)}")
    if index_name not in pinecone.list_indexes().names():
        raise ValueError(f"Index {index_name} not found in Pinecone")
    else:
        return pinecone.Index(index_name)


def query_index(
        index,
        query_text,
        model=None,
        tokenizer=None,
        top_k=5,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Query the Pinecone index with a text query and return relevant documents.

    Args:
        index: Pinecone Index object
        query_text: String containing the user's query
        model: Pre-trained embedding model (optional)
        tokenizer: Pre-trained tokenizer (optional)
        top_k: Number of top results to return
        device: Device to run the model on (cuda or cpu)

    Returns:
        List of dictionaries containing matched documents and their metadata
    """
    model_name = "jinaai/jina-embeddings-v3"

    if model is None:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # Use model_name, not model

    model.to(device)
    inputs = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().squeeze().tolist()

    try:
        results = index.query(vector=query_embeddings, top_k=top_k, include_metadata=True)
        return results["matches"]
    except PineconeException as e:
        raise Exception(f"Error encountered during query: {str(e)}")


def generate_responses(query_text, retrieved_docs):
    """
    Generate responses based on the user's query and retrieved documents.

    Args:
        query_text: String containing the user's query
        retrieved_docs: List of dictionaries containing matched documents and their metadata

    Returns:
        List of strings containing responses
    """
    groq_api = os.getenv("GROQ_API_KEY")
    if not groq_api:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    client = Groq(api_key=groq_api)

    context_parts = []
    for doc in retrieved_docs:
        text = doc["metadata"]["text"]
        url = doc["metadata"]["url"]
        title = doc["metadata"]["title"]
        chunk_number = doc["metadata"]["chunk_number"]
        score = doc["score"]
        context_entry = (
            f"document title: {title} \n"
            f"document text: {text}"
            f"document URL: {url} \n"
            f"chunk number: {chunk_number} \n"
            f"similarity score: {score}"

        )
        context_parts.append(context_entry)
    context = " \n-----\n".join(context_parts)
    prompt = (
        f"Question: {query_text}\n\n"
        f"Context (Retrieved Documents with Metadata):\n{context}\n\n"
        "Answer:"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": ("You're INSA Chatbot, a helpful chatbot assistant for insa Rennes students."
                         "Use the provided context, including metadata, to give accurate and detailed answers."
                         "if the context is not enough, you can ask for more information or clarification."
                         "if possible provide a link to the source of the information.")
             },
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.1,
        max_tokens=1000
    )

    return chat_completion.choices[0].message.content



def rag_search(query: str) -> str:
    """Searches the INSA Rennes knowledge base using RAG and generates a response."""
    index = connect_to_index()
    retrieved_docs = query_index(index, query)
    return generate_responses(query, retrieved_docs)

def timetable_lookup(query: str) -> str:
    """Fetches timetable information for INSA Rennes students."""
    return f"Timetable info for '{query}' (placeholder response)."