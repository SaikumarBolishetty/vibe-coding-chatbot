import os
from typing import List, Dict, Optional
import re
from dotenv import load_dotenv

load_dotenv()

class SimpleRAGService:
    def __init__(self):
        self.documents = []
        self._add_sample_documents()
        print("✅ Simple RAG service initialized with keyword-based search")
    
    def _add_sample_documents(self):
        """Add sample documents for testing"""
        self.documents = [
            {
                "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. It's designed to be easy to use and learn, with automatic interactive API documentation.",
                "metadata": "fastapi_intro"
            },
            {
                "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, automation, and many other fields.",
                "metadata": "python_intro"
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It includes algorithms like neural networks, decision trees, and support vector machines.",
                "metadata": "ml_intro"
            },
            {
                "content": "Docker is a platform for developing, shipping, and running applications in containers. Containers are lightweight, portable, and self-sufficient units that can run anywhere Docker is installed.",
                "metadata": "docker_intro"
            },
            {
                "content": "REST (Representational State Transfer) is an architectural style for designing networked applications. It uses HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs.",
                "metadata": "rest_api"
            },
            {
                "content": "Vector databases like Milvus are designed to store and search high-dimensional vector data efficiently. They're commonly used in AI applications for similarity search, recommendation systems, and semantic search.",
                "metadata": "vector_db"
            },
            {
                "content": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that context to generate more accurate and informative responses.",
                "metadata": "rag_concept"
            },
            {
                "content": "JavaScript is a programming language that is one of the core technologies of the World Wide Web. It enables interactive web pages and is an essential part of web applications.",
                "metadata": "javascript_intro"
            },
            {
                "content": "React is a JavaScript library for building user interfaces, particularly single-page applications. It's used for handling the view layer and can be used for developing both web and mobile applications.",
                "metadata": "react_intro"
            },
            {
                "content": "SQL (Structured Query Language) is a standard language for storing, manipulating, and retrieving data in relational database management systems.",
                "metadata": "sql_intro"
            }
        ]
        print(f"✅ Loaded {len(self.documents)} sample documents")
    
    def _calculate_similarity(self, query: str, content: str) -> float:
        """Simple keyword-based similarity calculation"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents based on keyword matching"""
        try:
            # Calculate similarity scores for all documents
            scored_docs = []
            for doc in self.documents:
                score = self._calculate_similarity(query, doc["content"])
                if score > 0:  # Only include documents with some similarity
                    scored_docs.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:top_k]
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """Get relevant context for a query"""
        similar_docs = self.search_similar_documents(query, top_k)
        
        if not similar_docs:
            return ""
        
        # Combine relevant documents into context
        context_parts = []
        for doc in similar_docs:
            context_parts.append(f"Document ({doc['metadata']}): {doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the knowledge base"""
        try:
            for doc in documents:
                self.documents.append({
                    "content": doc["content"],
                    "metadata": doc.get("metadata", "user_document")
                })
            print(f"✅ Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise e
    
    @property
    def num_entities(self):
        """Get the number of documents"""
        return len(self.documents)

# Global RAG service instance
rag_service = None

def get_rag_service() -> Optional[SimpleRAGService]:
    """Get or create RAG service instance"""
    global rag_service
    if rag_service is None:
        try:
            rag_service = SimpleRAGService()
        except Exception as e:
            print(f"⚠️  RAG service not available: {e}")
            return None
    return rag_service 