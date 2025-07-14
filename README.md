# NutriVibe - AI Nutrition Assistant

A powerful AI nutrition assistant built with FastAPI that provides interactive nutrition guidance with Retrieval-Augmented Generation (RAG) capabilities and **food image analysis**.

## Features

- 🎨 **Beautiful UI**: Modern, responsive nutrition assistant interface
- 🤖 **AI Nutrition Assistant**: Interactive AI powered by ChatGPT
- 📸 **Food Image Analysis**: Upload food images for nutrition info with OCR and AI vision
- 🔍 **RAG Integration**: Vector search with managed Milvus for enhanced responses
- 📱 **Mobile Responsive**: Works on all devices
- ⚡ **Fast Performance**: Built with FastAPI
- 📚 **Auto Documentation**: Interactive API docs
- 🗄️ **Vector Database**: Managed Milvus integration for semantic search

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **For ChatGPT Integration:**
   - Get an OpenAI API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
   - Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **For Image Analysis (Optional):**
   - Install Tesseract OCR for text extraction from images:
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```
   - The app will work without Tesseract, but OCR features will be limited

3. **For RAG Integration with Managed Milvus:**
   - Get your managed Milvus cluster credentials (host, port, user, password)
   - Add these to your `.env` file:
   ```
   MILVUS_HOST=your-managed-milvus-host
   MILVUS_PORT=your-managed-milvus-port
   MILVUS_USER=your-milvus-username
   MILVUS_PASSWORD=your-milvus-password
   MILVUS_SECURE=true
   ```

## Running the Application

### Method 1: Using Python directly
```bash
python main.py
```

### Method 2: Using uvicorn directly
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Accessing the Application

Once running, you can access:

- **Chatbot Interface**: http://localhost:8000/
- **Hello endpoint**: http://localhost:8000/hello
- **Chatbot API**: http://localhost:8000/ask (POST)
- **Document Management**: http://localhost:8000/documents (POST)
- **Document Search**: http://localhost:8000/documents/search (GET)
- **Document Count**: http://localhost:8000/documents/count (GET)
- **Image Analysis**: http://localhost:8000/analyze-image (POST)
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative API docs**: http://localhost:8000/redoc

## API Endpoints

- `GET /` - Serves the chatbot HTML interface
- `GET /hello` - Returns a greeting message
- `POST /ask` - Chatbot endpoint with RAG enhancement
- `POST /documents` - Add documents to vector database
- `GET /documents/search` - Search documents in vector database
- `GET /documents/count` - Get document count in vector database
- `POST /analyze-image` - Analyze food images for nutrition information

## Usage

1. Open http://localhost:8000/ in your browser
2. Type your nutrition questions in the chat interface
3. Upload food images using the "📸 Analyze Food Image" button
4. The assistant will respond with enhanced nutrition information using RAG

## Example API Usage

### Chat with RAG
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the macros in chicken breast?", "rag": true}'
```

### Analyze Food Image
```bash
curl -X POST "http://localhost:8000/analyze-image" \
     -F "image=@your_food_image.jpg"
```

### Add Documents
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your document content here", "metadata": "document_type"}'
```

### Search Documents
```bash
curl "http://localhost:8000/documents/search?query=python&top_k=3"
```

## RAG Features

- **Semantic Search**: Find relevant documents using vector similarity
- **Context Enhancement**: ChatGPT responses are enhanced with retrieved context
- **Document Management**: Add, search, and manage documents in the vector database
- **Sample Documents**: Pre-loaded with sample documents about programming topics

## Future Enhancements

- Add user authentication
- Save chat history
- Support for code syntax highlighting
- File upload for documents
- Multiple document collections 