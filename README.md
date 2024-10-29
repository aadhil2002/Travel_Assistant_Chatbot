
# Travel Assistant Chatbot 🌍🧳

A travel assistant chatbot that provides personalized travel recommendations using a Retrieval-Augmented Generation (RAG) approach. This project uses a vector database (Pinecone) for efficient document retrieval and Cohere for reranking, ensuring responses align with user preferences.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [Project Structure](#project-structure)
3. [Documentation](#documentation)
4. [Testing Instructions](#testing-instructions)

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/downloads) for version control
- A `.env` file containing API keys (details below)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/travel-assistant-chatbot.git
cd travel-assistant-chatbot
```

### Step 2: Install Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Create a `.env` file with the following keys (replacing placeholders with actual values):
```env
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### Step 4: Set Up Pinecone Index
To initialize the Pinecone vector index, run:
```bash
python rag.py
```

This script reads travel information from `dataset/travel_info.csv`, generates embeddings using a Sentence Transformer model, and upserts them into the Pinecone vector database.

---

## Project Structure

- `main.py` - Main application file for the chatbot interface (Streamlit-based).
- `rag.py` - Script to initialize and populate the vector database (Pinecone) with travel data.
- `dataset/` - Folder containing travel data (e.g., `travel_info.csv`).
- `.env` - Environment file for storing API keys (excluded in `.gitignore`).
- `requirements.txt` - Python dependencies required for this project.

---

## Documentation

### Approach
The chatbot uses a Retrieval-Augmented Generation (RAG) approach. The process involves:

1. **Query Embedding** - The user query is embedded using `SentenceTransformer` to facilitate semantic similarity matching.
2. **Document Retrieval** - The Pinecone vector database returns relevant documents based on the query embedding.
3. **Reranking with Cohere** - Retrieved documents are reranked using Cohere’s reranking model to align with user preferences.
4. **Response Generation** - Using Groq, a structured response is generated based on reranked documents and user preferences.

### Vector Database Selection
We selected **Pinecone** for fast and scalable vector-based search due to:
- Efficient indexing for semantic search with cosine similarity.
- Scalability and serverless infrastructure, allowing easy cloud integration.

### Architecture
The system is built on the following stack:

- **Streamlit** - Provides a web interface for interaction.
- **Pinecone** - Handles vector similarity search for retrieval.
- **Cohere** - Reranks documents to refine responses.
- **Groq** - Generates structured conversational responses based on filtered documents and user query.

---

## Testing Instructions

To test the chatbot locally, follow these steps:

1. **Start the Chatbot**
   Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. **Chatbot Interaction**
   In the browser, enter queries related to travel destinations, accommodations, or activities. The chatbot provides recommendations based on the travel preferences set in the sidebar.

3. **Example Queries**
   - “Suggest a budget-friendly travel destination for a 1-week vacation.”
   - “Where can I go for an adventure trip with luxury accommodation?”

4. **Additional Options**
   Use the sidebar to modify travel preferences (price range, themes, duration, accommodation type) and view how responses are adjusted based on these changes.

---
