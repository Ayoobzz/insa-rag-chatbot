import json
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException  # Updated import
from nlp import chunking


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


def query_index(
        index, query_text,
        model="jinaai/jina-embeddings-v3",
        tokenizer=None,
        topk=5,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Args:
        index: Pinecone index object
        query_text: String containing the user's query
        tokenizer: Pre-trained tokenizer for the embedding model
        model: Pre-trained embedding model
        top_k: Number of top results to return
        device: Device to run the model on (cuda or cpu)

    Returns:
        List of dictionaries containing matched documents and their metadata
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    inputs = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().squeeze().tolist()

    try:
        results = index.query(
            vector=query_embeddings,
            top_k=topk,
            include_metadata=True
        )
    except PineconeException as e:
        raise Exception(f"Error encountered during query: {str(e)}")

    return results["matches"]












"""def main():

    data_path = "../data/processed/cleaned.json"
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file {data_path}")


    try:
        data = chunking.chunk_data(data)
    except NameError:
        print("Warning: Proceeding without chunking as chunking module is not available")
    except Exception as e:
        raise Exception(f"Error in chunking data: {str(e)}")

    index_init(data)


main()"""