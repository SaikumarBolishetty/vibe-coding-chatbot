from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import os
from openai import OpenAI
from typing import Optional
from rag_service import get_rag_service

# Create FastAPI instance
app = FastAPI(title="Vibe AI Assistant with RAG", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for request
class ChatRequest(BaseModel):
    message: str

class DocumentRequest(BaseModel):
    content: str
    metadata: str = ""

# Initialize OpenAI client (will be None if no API key)
openai_client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        print("✅ OpenAI API key found - ChatGPT integration enabled!")
    else:
        print("⚠️  No OpenAI API key found - using demo responses")
except Exception as e:
    print(f"⚠️  Error initializing OpenAI: {e} - using demo responses")

# Initialize RAG service
rag_service = None
try:
    rag_service = get_rag_service()
    if rag_service:
        print("✅ Simple RAG service initialized - Keyword-based search enabled!")
    else:
        print("⚠️  RAG service not available - using basic responses")
except Exception as e:
    print(f"⚠️  Error initializing RAG service: {e} - using basic responses")

@app.get("/")
async def root():
    """Serve the chatbot HTML page"""
    return FileResponse("static/index.html")

@app.get("/hello")
async def hello():
    """Hello endpoint"""
    return {"message": "Hello from Vibe Coding Chatbot!"}

def get_demo_response(user_message: str) -> str:
    """Get demo response when OpenAI API is not available"""
    user_message = user_message.lower()
    
    if "hello" in user_message or "hi" in user_message:
        return "Hello! I'm your AI coding assistant. How can I help you with your coding questions today?"
    elif "python" in user_message:
        return "Python is a versatile programming language! It's great for web development, data science, AI, and automation. What specific Python question do you have?"
    elif "javascript" in user_message or "js" in user_message:
        return "JavaScript is the language of the web! It's used for frontend development, backend (Node.js), and mobile apps. What would you like to know about JavaScript?"
    elif "fastapi" in user_message:
        return "FastAPI is a modern, fast web framework for building APIs with Python! It's based on standard Python type hints and provides automatic API documentation. It's perfect for building high-performance APIs."
    elif "help" in user_message:
        return "I can help you with coding questions, explain programming concepts, debug issues, and provide code examples. Just ask me anything about programming!"
    elif "thank" in user_message:
        return "You're welcome! I'm here to help with all your coding questions. Feel free to ask anything!"
    else:
        return f"I understand you're asking about: '{user_message}'. This is a demo response. To get real ChatGPT responses, please set your OPENAI_API_KEY environment variable."

async def get_chatgpt_response(user_message: str, context: str = "") -> str:
    """Get response from ChatGPT API with optional RAG context"""
    if not openai_client:
        return "OpenAI client not initialized"
    
    try:
        # Prepare system message with context if available
        system_content = "You are a helpful AI assistant. You can help with programming, general questions, explanations, and various topics. Be friendly, informative, and provide helpful responses."
        
        if context:
            system_content += f"\n\nUse the following context to provide more accurate and detailed answers:\n{context}"
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=800,
            temperature=0.7
        )
        content = response.choices[0].message.content
        return content if content else "No response from ChatGPT"
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return f"Sorry, I encountered an error while connecting to ChatGPT: {str(e)}"

@app.post("/ask")
async def ask_chatbot(request: ChatRequest):
    """Chatbot endpoint with RAG-enhanced ChatGPT responses"""
    try:
        user_message = request.message
        
        # Get relevant context from vector database if RAG is available
        context = ""
        if rag_service:
            try:
                context = rag_service.get_context_for_query(user_message, top_k=3)
                if context:
                    print(f"🔍 Found relevant context for query: '{user_message[:50]}...'")
            except Exception as e:
                print(f"⚠️  Error retrieving context: {e}")
        
        # Use ChatGPT if available, otherwise use demo responses
        if openai_client:
            response = await get_chatgpt_response(user_message, context)
        else:
            response = get_demo_response(user_message)
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/documents")
async def add_document(request: DocumentRequest):
    """Add a document to the vector database"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        rag_service.add_documents([{
            "content": request.content,
            "metadata": request.metadata
        }])
        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.get("/documents/search")
async def search_documents(query: str, top_k: int = 3):
    """Search documents in the vector database"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        results = rag_service.search_similar_documents(query, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/documents/count")
async def get_document_count():
    """Get the number of documents in the knowledge base"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        count = rag_service.num_entities
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document count: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 