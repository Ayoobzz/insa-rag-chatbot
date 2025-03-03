import json
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from pinecone.openapi_support.exceptions import PineconeApiException

from nlp import chunking


def index_init(data, index_name="insa-chatbot"):
    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Initialize Pinecone
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index exists or create it
    try:
        # Attempt to create the index
        pinecone.create_index(
            index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {index_name}")
    except PineconeApiException as e:
        if e.status == 409:  # Index already exists
            print(f"Index {index_name} already exists, connecting to it.")
        else:
            raise  # Re-raise other unexpected errors

    # Connect to the index (whether newly created or existing)
    index = pinecone.Index(index_name)

    # Load the embedding model
    model_name = "jinaai/jina-embeddings-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Prepare data for upserting
    to_upsert = []
    for key, val in data.items():
        url = key
        texts = val[0]
        title = val[1]

        # Tokenize and compute embeddings
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract `[CLS]` token representation
            embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

        # Upsert data into Pinecone
        to_upsert.append((url, embeddings.squeeze().tolist(), {"title": title, "text": texts, "url": url}))

    # Perform batch upsert to Pinecone
    if to_upsert:
        index.upsert(vectors=to_upsert)

    print(f"Successfully indexed {len(to_upsert)} items into {index_name}.")

# Load data and initialize the index
with open("../data/processed/cleaned.json", encoding="utf-8") as f:
    data = json.load(f)
    data=chunking.chunk_data(data)
    index_init(data)