# rag_database_setup.py
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Load the CSV file
file_path = r"dataset\travel_info.csv"
df = pd.read_csv(file_path)

# Initialize the Sentence Transformer model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "travel-assistant-db"

if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Prepare data for indexing
for _, row in df.iterrows():
    # Create the text content that will be used for retrieval
    text_content = f"Destination: {row['Destination Name']}, Location: {row['Country/City']}, " \
           f"Accommodation: {row['Accommodation Options']}, Activities: {row['Activities/Attractions']}, " \
           f"Package: {row['Travel Packages']}, Price:{row['Price']}," \
           f"Availability:{row['Availability']}, Reviews: {row['Ratings/Reviews']}"
    
    embedding = model.encode(text_content)
    
    metadata = {
        "text": text_content  # Only include the text field in metadata
    }
    
    # Upsert to Pinecone
    index.upsert([(str(_), embedding, metadata)])

print("Data upserted to Pinecone index.")