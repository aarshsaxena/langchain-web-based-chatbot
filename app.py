from flask import Flask, request, jsonify
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize session variables
session_data = {
    "vector_store": None,
    "chat_history": []
}

# Function to load data from URL and create vector store
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create a vector store from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    return vector_store

# Function to create a context-aware retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

# Function to create the conversational RAG (Retrieval-Augmented Generation) chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# API Endpoint: Load data and initialize vector store
@app.route('/initialize', methods=['POST'])
def initialize():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        session_data['vector_store'] = get_vectorstore_from_url(url)
        return jsonify({'message': 'Vector store initialized successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API Endpoint: Handle conversation input
@app.route('/conversation', methods=['POST'])
def conversation():
    data = request.get_json()
    user_input = data.get('input')

    if not user_input:
        return jsonify({'error': 'User input is required'}), 400

    if session_data['vector_store'] is None:
        return jsonify({'error': 'Vector store is not initialized'}), 400

    # Get the context retriever and conversational RAG chain
    retriever_chain = get_context_retriever_chain(session_data['vector_store'])
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    try:
        # Get response from the conversation RAG chain
        response = conversation_rag_chain.invoke({
            "chat_history": session_data['chat_history'],
            "input": user_input
        })

        # Append user input and model response to chat history
        session_data['chat_history'].append(HumanMessage(content=user_input))
        session_data['chat_history'].append(AIMessage(content=response['answer']))

        return jsonify({'response': response['answer']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
