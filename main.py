import os
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import cohere
from groq import Groq
import numpy as np

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

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

# Streamlit interface setup with emojis
st.title("🌍 Travel Chatbot 🧳")
st.write("Ask about travel destinations, accommodations, activities, or packages! ✈️🌴")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {"price_range": (0, 1000)}
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Sidebar with preferences and clear chat button
st.sidebar.header("🎒 Your Preferences")
price_range = st.sidebar.slider("💵 Select Price Range", 0, 5000, (100, 1000))
st.session_state.user_preferences["price_range"] = price_range

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.session_state.user_input = ""

def get_query_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

def create_chat_prompt(context_entries, user_query, preferences):
    """Create an optimized prompt for better travel recommendations"""
    # Format context entries more concisely
    formatted_context = "\n".join([
        f"[Option {i+1}]\n{entry['text']}"
        for i, entry in enumerate(context_entries)
    ])
    
    return f"""You are a knowledgeable travel assistant. Provide personalized travel recommendations based on the following information.

Query: {user_query}

Budget: ${preferences['price_range'][0]} - ${preferences['price_range'][1]}

Available Options:
{formatted_context}

Instructions:
1. Focus on options within the user's budget range
2. Highlight key features:
   - Location highlights
   - Accommodation details
   - Must-see attractions
   - Special offers or packages
3. If recommending multiple options, organize them by best value
4. Include specific prices when available
5. Be concise but informative
6. If no perfect matches exist, suggest closest alternatives

Format your response in a clear, organized manner with appropriate sections."""

def process_chat():
    if st.session_state.user_input:
        user_input = st.session_state.user_input
        
        # Embed the query
        question_embedding = get_query_embedding(user_input)
        
        # Add price range to query
        price_range_text = f"Price Range: ${st.session_state.user_preferences['price_range'][0]} - ${st.session_state.user_preferences['price_range'][1]}"
        query_with_preferences = f"{price_range_text} {user_input}".strip()
        
        # Query Pinecone
        query_result = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract and process documents
        doc_texts = [match.metadata.get("text", "") for match in query_result.matches]
        docs = {text: i for i, text in enumerate(doc_texts) if text}
        
        if not docs:
            st.error("🚫 No relevant results found. Please try a different query.")
            return
        
        # Rerank documents
        rerank_docs = co.rerank(
            model="rerank-english-v3.0",
            query=query_with_preferences,
            documents=list(docs.keys()),
            top_n=5,
            return_documents=True
        )
        
        # Prepare context and generate response
        reranked_texts = [doc.document.text for doc in rerank_docs.results]
        context_entries = [{"text": text} for text in reranked_texts]
        prompt = create_chat_prompt(context_entries, user_input, st.session_state.user_preferences)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.2-90b-text-preview"
        )
        
        bot_response = chat_completion.choices[0].message.content
        
        # Update chat history
        st.session_state.history.append(("User", user_input))
        st.session_state.history.append(("Bot", bot_response))
        
        # Clear the input after processing
        st.session_state.user_input = ""

# Display chat history
for speaker, text in st.session_state.history:
    if speaker == "User":
        st.write(f"🧍 You: {text}")
    else:
        st.write(f"🤖 Bot: {text}")

# Input field at the bottom
st.text_input(
    "💬 You: ",
    key="user_input",
    on_change=process_chat,
    value=st.session_state.user_input
)