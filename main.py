import os
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from dotenv import load_dotenv
import cohere

# Load environment variables
load_dotenv()

# Initialize API keys
@st.cache_resource
def init_api_keys():
    return {
        'pinecone': Pinecone(api_key=os.getenv("PINECONE_API_KEY")),
        'groq': Groq(api_key=os.getenv("GROQ_API_KEY")),
        'model': SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True),
        'cohere': cohere.Client(os.getenv("COHERE_API_KEY"))
    }

api_keys = init_api_keys()
index = api_keys['pinecone'].Index("travel-assistant-db")

# Streamlit interface
st.title("🌍 Travel Chatbot 🧳")
st.write("Ask about travel destinations! ✈️🌴")

# Simplified preferences
with st.sidebar:
    st.header("🎒 Quick Preferences")
    
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
        st.session_state.conversation_context = []
        st.rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preferences_history" not in st.session_state:
    st.session_state.preferences_history = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

def get_query_embedding(text):
    return api_keys['model'].encode(text).tolist()



# Prompt template
def create_chat_prompt(context_entries, user_query, conversation_history):
    """Enhanced prompt creation with conversation context"""
    context = "\n".join([entry['text'] for entry in context_entries[:3]])
    
    # Create conversation history string
    conversation_context = ""
    if conversation_history:
        conversation_context = "Previous conversation:\n" + "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
            for msg in conversation_history[-4:]
        ]) + "\n\n"
    
    return f"""{conversation_context}Create a travel recommendation based on an imaginary world setting:

Query: {user_query}

User Preferences:
- Themes: {', '.join(themes) if themes else 'Any'}
- Duration: {duration}
- Accommodation: {accommodation}

Available Options:
{context}

Instructions:
1. Present recommendations that:
   - Consider the previous conversation context when relevant
   - Are consistent with any previously discussed destinations
   - Address any follow-up questions about previously mentioned places
   - Present new information while maintaining context

2. Format guidelines:
   - Present each destination as a separate section
   - Include only factual, verified information
   - Remove any reference numbers or internal notes
   - Don't include phrases like "Option X" or "Alternative"
   - Don't include any metadata or processing notes,or Note: and () statements.
   
3. Response structure:
   - Start with a brief introduction
   - List main destinations with detailed breakdowns
   - End with relevant practical tips
   
4. Keep the response:
   - Focused on user's preferences
   - Aligned with selected themes
   - Appropriate for requested duration
   - Clear and professional
   - Consistent with previous recommendations
   - When filtering out locations,based on continents,just look only at the country and not at the location.
   - Every location is not based on real location and is just imaginary
   - Always respond in a positive manner
   - Always focus on the previous conversation while provide the response
   - Structure the response in such a way that the top one is the most affordable,and the bottom one is the most expensive
   

5. Additional requirements:
   - Be direct and concise
   - Suggest viable alternatives if perfect matches aren't available"""

def rerank_results(query, docs, top_k=3):
    """
    Rerank the search results using Cohere's reranking capability
    """
    try:
        # Extract text from docs
        texts = [doc["text"] for doc in docs]
        
        # Use Cohere's rerank endpoint
        reranked = api_keys['cohere'].rerank(
            query=query,
            documents=texts,
            top_n=top_k,
            model='rerank-english-v2.0'
        )
        
        # Create new docs list with reranked results
        reranked_docs = []
        for result in reranked:
            # Access the index as an integer attribute
            doc_index = int(result.index)
            reranked_docs.append({
                "text": texts[doc_index],
                "relevance_score": float(result.relevance_score)
            })
        
        return reranked_docs
    except Exception as e:
        print(f"Error in reranking: {str(e)}")
        # Fallback to original docs if reranking fails
        return docs[:top_k]

def get_response(user_input):
    try:
        # Get query embedding
        embedding = get_query_embedding(user_input)
        
        # Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
        
        if not results.matches:
            return "I couldn't find relevant information. Please try a different query."
        
        # Process initial results
        docs = [{"text": match.metadata.get("text", "")} for match in results.matches]
        
        # Rerank results using Cohere
        reranked_docs = rerank_results(user_input, docs)
        
        if not reranked_docs:
            reranked_docs = docs[:3]
        
        # Generate response with conversation context
        prompt = create_chat_prompt(reranked_docs, user_input, st.session_state.conversation_context)
        response = api_keys['groq'].chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.2-90b-text-preview"
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return "I encountered an error processing your request. Please try again."

def save_current_preferences():
    """Save current preferences to history"""
    return {
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
            
            if idx < len(st.session_state.preferences_history):
                with st.expander("View preferences at time of query"):
                    prefs = st.session_state.preferences_history[idx]
                    st.write(f"Themes: {', '.join(prefs['themes'])}")
                    st.write(f"Duration: {prefs['duration']}")
                    st.write(f"Accommodation: {prefs['accommodation']}")

# Chat input
if prompt := st.chat_input("💬 Ask me anything about travel:"):
    # Save current preferences before processing the message
    current_prefs = save_current_preferences()
    
    # Add user message to conversation context and history
    user_message = {"role": "user", "content": prompt}
    st.session_state.conversation_context.append(user_message)
    st.session_state.messages.append(user_message)
    st.session_state.preferences_history.append(current_prefs)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.write(response)
            
            # Add assistant response to conversation context and history
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.conversation_context.append(assistant_message)
            st.session_state.messages.append(assistant_message)
            st.session_state.preferences_history.append(current_prefs)