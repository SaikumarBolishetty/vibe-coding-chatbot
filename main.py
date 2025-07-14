from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import os
import io
import base64
from PIL import Image
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not installed - OCR features will be limited")
from openai import OpenAI
from typing import Optional
from rag_service import get_rag_service

# Create FastAPI instance
app = FastAPI(title="Vibe AI Assistant with RAG", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for request
class ChatRequest(BaseModel):
    question: str
    rag: bool = True

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
        return "Hello! I'm NutriVibe, your AI nutrition assistant. How can I help you with your nutrition and health questions today?"
    elif "macro" in user_message or "protein" in user_message or "carb" in user_message or "fat" in user_message:
        if "chicken" in user_message and "sandwich" in user_message:
            return "A typical chicken sandwich contains approximately:\n\n🍗 **Protein**: 25-35g\n🍞 **Carbs**: 30-45g (from bread)\n🥑 **Fat**: 8-15g\n📊 **Total Calories**: 350-500\n\n*Note: These are approximate values. Actual macros depend on the specific ingredients, portion size, and preparation method. For accurate tracking, I recommend checking the nutrition label or using a food database app.*"
        else:
            return "I can help you with macronutrient information! Please be specific about what food you'd like to know about. For example: 'What are the macros in chicken sandwich?' or 'How much protein is in salmon?'"
    elif "calorie" in user_message or "calories" in user_message:
        return "I can help you with calorie information! Please specify what food you're asking about, and I'll provide detailed nutritional information."
    elif "recipe" in user_message:
        return "I'd love to help you with healthy recipes! What type of recipe are you looking for? (e.g., high-protein meals, low-carb options, vegetarian dishes, etc.)"
    elif "healthy" in user_message or "nutrition" in user_message:
        return "Great question about nutrition! I can help you with meal planning, understanding food labels, macronutrients, and healthy eating tips. What specific aspect would you like to know more about?"
    elif "help" in user_message:
        return "I'm NutriVibe, your AI nutrition assistant! I can help you with:\n• Macronutrient information\n• Calorie counting\n• Healthy recipes\n• Nutrition advice\n• Meal planning\n\nJust ask me anything about nutrition and healthy eating!"
    elif "thank" in user_message:
        return "You're welcome! I'm here to help with all your nutrition questions. Feel free to ask anything about healthy eating, recipes, or nutritional information!"
    else:
        return f"I understand you're asking about: '{user_message}'. This is a demo response. To get more detailed nutrition information, please set your OPENAI_API_KEY environment variable for full ChatGPT integration."

async def get_chatgpt_response(user_message: str, context: str = "") -> str:
    """Get response from ChatGPT API with optional RAG context"""
    if not openai_client:
        return "OpenAI client not initialized"
    
    try:
        # Prepare system message with context if available
        system_content = "You are NutriVibe, a helpful AI nutrition assistant. You specialize in nutrition, healthy eating, meal planning, and providing accurate nutritional information. Be friendly, informative, and provide helpful responses about food, nutrition, and health. Always provide practical, evidence-based advice."
        
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
    """Chatbot endpoint with RAG-enhanced ChatGPT responses and UI metadata"""
    try:
        user_message = request.question
        context = ""
        rag_used = False
        sources = []
        similar_docs = []
        
        # Only use RAG if enabled and service is available
        if request.rag and rag_service:
            try:
                similar_docs = rag_service.search_similar_documents(user_message, top_k=3)
                context = "\n\n".join([f"Document ({doc['metadata']}): {doc['content']}" for doc in similar_docs])
                if context.strip():
                    rag_used = True
                    sources = [doc['metadata'] for doc in similar_docs if doc.get('metadata')]
                    print(f"🔍 Found relevant context for query: '{user_message[:50]}...' (RAG used)")
            except Exception as e:
                print(f"⚠️  Error retrieving context: {e}")
        
        # Use ChatGPT if available, otherwise use demo responses
        if openai_client:
            response = await get_chatgpt_response(user_message, context)
        else:
            response = get_demo_response(user_message)
        
        return {"answer": response, "rag_used": rag_used, "sources": sources}
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

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        # Open image using PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using pytesseract (if available)
        if TESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(image)
                return (text or "").strip()
            except Exception as e:
                print(f"OCR error: {e}")
                return ""
        else:
            return ""
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

def analyze_food_image(image_bytes: bytes) -> str:
    """Analyze food image and return nutrition information"""
    try:
        # Extract text from image (nutrition labels, etc.)
        extracted_text = extract_text_from_image(image_bytes)
        
        # Convert image to base64 for OpenAI vision API
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Use OpenAI Vision API if available
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are NutriVibe, a nutrition expert. Analyze this food image and provide detailed nutritional information. If you can see a nutrition label, extract the exact values. If not, estimate based on the food items visible. Be specific about calories, protein, carbs, fat, and other relevant nutrients. Format your response clearly with emojis and bullet points."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Please analyze this food image and provide detailed nutritional information. If there's text in the image, here's what I extracted: '{extracted_text}'. Use this information along with visual analysis to provide accurate nutrition details."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error with OpenAI Vision API: {e}")
                # Fall back to text-based analysis
        
        # Fallback analysis based on extracted text
        if extracted_text:
            return f"📸 **Image Analysis Results**\n\nI found the following text in your image:\n\n```\n{extracted_text}\n```\n\nTo get detailed nutrition analysis, please set your OPENAI_API_KEY environment variable for full image recognition capabilities."
        else:
            return "📸 **Image Analysis Results**\n\nI couldn't extract any text from your image. To get detailed nutrition analysis with food recognition, please set your OPENAI_API_KEY environment variable for full image recognition capabilities."
            
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"

@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """Analyze food image and return nutrition information"""
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Analyze the image
        analysis = analyze_food_image(image_bytes)
        
        return {"analysis": analysis}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 