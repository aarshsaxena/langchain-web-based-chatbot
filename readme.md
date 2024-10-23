# Conversational AI with Langchain and Flask

This project is a conversational AI system built using Langchain and Flask. It allows users to retrieve context-aware responses from a knowledge base created from web documents by splitting and vectorizing the content. The system uses OpenAI embeddings and Chroma for vector storage.

## Features

- **Web Document Loader**: Load content from a given URL and split it into manageable document chunks.
- **Vector Store**: Utilize Chroma to store document embeddings for fast and efficient retrieval.
- **Conversation Retriever**: Retrieve context-aware responses based on previous conversation history.
- **Conversational Chain**: Combines the retrieval chain with an LLM (ChatGPT) to generate human-like responses.
- **Flask API**: Expose the system as an API to handle conversations.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- Flask
- Langchain
- OpenAI API Key
- ChromaDB
- Dotenv

## Installation

1. Clone the repository:

   git clone https://github.com/aarshsaxena/langchain-web-based-chatbot.git
   cd conversational-ai-flask

2. Install Dependencies

pip install -r requirements.txt

3. Set up environment variables: Create a .env file in the root directory and add your OpenAI API key:

OPENAI_API_KEY=your-openai-api-key

4. Run the Flask API:

python app.py



## API Endpoints

1. /initialize
Method: POST
Description: Initialize the vector store by loading content from a provided URL.

Request Body:

{
  "url": "https://example.com"
}

Response:

{
  "message": "Vector store initialized"
}

2. /conversation
Method: POST
Description: Handles user input and returns context-aware AI responses.
Request Body:

{
  "input": "User's question or statement",
  "chat_history": "Previous conversation history (optional)"
}
Response:

{
  "response": "AI's response"
}
