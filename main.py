import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import cohere
from groq import Groq
import numpy as np

# Initialize clients
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("travel-assistant-db")
groq_client = Groq(api_key=GROQ_API_KEY)

# Load the embedding model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Streamlit interface setup
st.title("Travel Chatbot")
st.write("Ask about travel destinations, accommodations, activities, or packages!")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {"budget": None}

# User input and preferences
user_input = st.text_input("You: ", "")
st.sidebar.header("Your Preferences")
budget = st.sidebar.selectbox("Budget", ["Any", "Affordable", "Luxury"])

if budget != "Any":
    st.session_state.user_preferences["budget"] = budget

def get_query_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

# Chat functionality
if user_input:
    # Embed the query
    question_embedding = get_query_embedding(user_input)
    
    # Add budget preference to query if specified
    preference_text = ""
    if st.session_state.user_preferences["budget"]:
        preference_text = f"Budget: {st.session_state.user_preferences['budget']}"
    
    query_with_preferences = f"{preference_text} {user_input}".strip()
    
    # Query Pinecone
    query_result = index.query(
        vector=question_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Extract text from metadata for reranking
    doc_texts = [match.metadata.get("text", "") for match in query_result.matches]
    docs = {text: i for i, text in enumerate(doc_texts) if text}
    
    if not docs:
        st.error("No relevant results found. Please try a different query.")
    else:
        # Rerank documents
        rerank_docs = co.rerank(
            model="rerank-english-v3.0",
            query=query_with_preferences,
            documents=list(docs.keys()),
            top_n=5,
            return_documents=True
        )
        
        # Prepare context
        reranked_texts = [doc.document.text for doc in rerank_docs.results]
        context = " ".join(reranked_texts)
        
        # Generate response
        template = f"Based on the following context: {context}, generate a response for: {user_input}. Keep user preferences in mind: {preference_text}"
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": template}
            ],
            model="llama-3.2-90b-text-preview"
        )
        
        bot_response = chat_completion.choices[0].message.content
        
        # Update chat history
        st.session_state.history.append(("User", user_input))
        st.session_state.history.append(("Bot", bot_response))

# Display chat history
for speaker, text in st.session_state.history:
    if speaker == "User":
        st.write(f"You: {text}")
    else:
        st.write(f"Bot: {text}")