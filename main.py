import os
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import cohere
from groq import Groq
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients only once at startup
@st.cache_resource
def init_clients():
    return {
        'cohere': cohere.Client(os.getenv("COHERE_API_KEY")),
        'pinecone': Pinecone(api_key=os.getenv("PINECONE_API_KEY")),
        'groq': Groq(api_key=os.getenv("GROQ_API_KEY")),
        'model': SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    }

clients = init_clients()
index = clients['pinecone'].Index("travel-assistant-db")

# Streamlit interface
st.title("🌍 Travel Chatbot 🧳")
st.write("Ask about travel destinations! ✈️🌴")

# Simplified preferences
with st.sidebar:
    st.header("🎒 Quick Preferences")
    
    # Simplified price range with predefined options
    price_ranges = {
        "Budget ($0-$1500)": (0, 1500),
        "Moderate ($1500-$2500)": (1500, 2500),
        "Luxury ($2500+)": (2500, 5000)
    }
    selected_range = st.selectbox("💵 Budget", list(price_ranges.keys()), key='budget')
    
    # Limit theme selection to 3 choices
    themes = st.multiselect(
        "🎯 Top 3 Themes",
        ["Adventure", "Relaxation", "Cultural", "Food", "Nature", "Urban", "Beach"],
        max_selections=3,
        key='themes'
    )
    
    duration = st.selectbox(
        "⏱️ Duration",
        ["1-7 days", "1-2 weeks", "2-3 weeks", "1 month+"],
        key='duration'
    )
    
    accommodation = st.selectbox(
        "🏨 Accommodation",
        ["Any", "Hotel", "Resort", "Rental", "Hostel", "Camping"],
        key='accommodation'
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.preferences_history = []
        st.experimental_rerun()

# Initialize chat history and preferences history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preferences_history" not in st.session_state:
    st.session_state.preferences_history = []

def get_query_embedding(text):
    return clients['model'].encode(text).tolist()

def create_chat_prompt(context_entries, user_query):
    """Simplified prompt creation"""
    price_range = price_ranges[selected_range]
    context = "\n".join([entry['text'] for entry in context_entries[:3]])
    
    return f"""Create a travel recommendation based on:

Query: {user_query}

User Preferences:
- Budget: ${price_range[0]}-${price_range[1]}
- Themes: {', '.join(themes) if themes else 'Any'}
- Duration: {duration}
- Accommodation: {accommodation}

Available Options:
{context}

Instructions:
1. Present recommendations in a clear, direct format focusing on:
   - Destination name and country
   - Location highlights and unique features
   - Accommodation details matching user's preferences
   - Must-see attractions
   - Special offers or packages with specific prices
   
2. Format guidelines:
   - Present each destination as a separate section
   - Include only factual, verified information
   - Remove any reference numbers, option numbers, or internal notes
   - Don't mention data sources or adjustments
   - Don't include phrases like "Option X" or "Alternative"
   - Don't include any metadata or processing notes
   
3. Response structure:
   - Start with a brief introduction
   - List main destinations with detailed breakdowns
   - Include specific prices where available
   - End with relevant practical tips
   
4. Keep the response:
   - Focused on user's preferences
   - Within specified budget range
   - Aligned with selected themes
   - Appropriate for requested duration
   - Clear and professional
   - Also when try to filter out the locations based on the continent,look at the country and filter,and not by looking at the place or location.
   - Even if the location is incorrect,try to make the response based on the incorrect location,and do not try to correct the incorrect location
   - Also do not include statements like this in the response: (note: this seems to be an incorrect location, but I've tried to make the response based on it)
   - Always respond in a positive manner

5. Additional requirements:
   - Include exact prices when available
   - Be direct and concise
   - Suggest viable alternatives if perfect matches aren't available"""

def get_response(user_input):
    # Get query embedding
    embedding = get_query_embedding(user_input)
    
    # Query Pinecone with reduced top_k
    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )
    
    if not results.matches:
        return "I couldn't find relevant information. Please try a different query."
    
    # Process results
    docs = [{"text": match.metadata.get("text", "")} for match in results.matches]
    
    # Generate response
    prompt = create_chat_prompt(docs, user_input)
    response = clients['groq'].chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.2-90b-text-preview"
    )
    
    return response.choices[0].message.content

def save_current_preferences():
    """Save current preferences to history"""
    return {
        'budget': selected_range,
        'themes': themes.copy() if themes else [],
        'duration': duration,
        'accommodation': accommodation
    }

# Create a container for the chat history
chat_container = st.container()

# Display chat history with associated preferences
with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # If this message has associated preferences, show them in an expander
            if idx < len(st.session_state.preferences_history):
                with st.expander("View preferences at time of query"):
                    prefs = st.session_state.preferences_history[idx]
                    st.write(f"Budget: {prefs['budget']}")
                    st.write(f"Themes: {', '.join(prefs['themes'])}")
                    st.write(f"Duration: {prefs['duration']}")
                    st.write(f"Accommodation: {prefs['accommodation']}")

# Chat input
if prompt := st.chat_input("💬 Ask me anything about travel:"):
    # Save current preferences before processing the message
    current_prefs = save_current_preferences()
    
    # Add user message and preferences to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.preferences_history.append(current_prefs)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.write(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Add current preferences again for the assistant's response
            st.session_state.preferences_history.append(current_prefs)