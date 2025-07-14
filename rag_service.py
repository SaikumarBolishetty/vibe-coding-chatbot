import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer

load_dotenv()

class MilvusRAGService:
    def __init__(self):
        self.collection_name = "documents"
        # Use a large embedding model (bge-large-en-v1.5, 1024 dimensions)
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_dim = 1024  # bge-large-en-v1.5 has 1024 dimensions
        self._connect_to_milvus()
        self._setup_collection()
        self._add_sample_documents()

    def _connect_to_milvus(self):
        host = os.getenv("MILVUS_HOST")
        port = os.getenv("MILVUS_PORT")
        user = os.getenv("MILVUS_USER")
        password = os.getenv("MILVUS_PASSWORD")
        secure = os.getenv("MILVUS_SECURE", "true").lower() == "true"
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                user=user,
                password=password,
                secure=secure
            )
            print("✅ Connected to managed Milvus successfully!")
        except Exception as e:
            print(f"⚠️  Could not connect to Milvus: {e}")
            raise e

    def _setup_collection(self):
        if utility.has_collection(self.collection_name):
            print(f"✅ Collection '{self.collection_name}' already exists")
            self.collection = Collection(self.collection_name)
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            # Update dimension to 1024 for bge-large-en-v1.5
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
        ]
        schema = CollectionSchema(fields=fields, description="Document embeddings for RAG")
        self.collection = Collection(name=self.collection_name, schema=schema)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ Created collection '{self.collection_name}' with index (dim={self.embedding_dim})")

    def _add_sample_documents(self):
        if self.collection.num_entities > 0:
            print("✅ Collection already has documents")
            return
        sample_docs = [
            {"content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. It's designed to be easy to use and learn, with automatic interactive API documentation.", "metadata": "fastapi_intro"},
            {"content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, automation, and many other fields.", "metadata": "python_intro"},
            {"content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It includes algorithms like neural networks, decision trees, and support vector machines.", "metadata": "ml_intro"},
            {"content": "Docker is a platform for developing, shipping, and running applications in containers. Containers are lightweight, portable, and self-sufficient units that can run anywhere Docker is installed.", "metadata": "docker_intro"},
            {"content": "REST (Representational State Transfer) is an architectural style for designing networked applications. It uses HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs.", "metadata": "rest_api"},
            {"content": "Vector databases like Milvus are designed to store and search high-dimensional vector data efficiently. They're commonly used in AI applications for similarity search, recommendation systems, and semantic search.", "metadata": "vector_db"},
            {"content": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that context to generate more accurate and informative responses.", "metadata": "rag_concept"},
            {"content": "JavaScript is a programming language that is one of the core technologies of the World Wide Web. It enables interactive web pages and is an essential part of web applications.", "metadata": "javascript_intro"},
            {"content": "React is a JavaScript library for building user interfaces, particularly single-page applications. It's used for handling the view layer and can be used for developing both web and mobile applications.", "metadata": "react_intro"},
            {"content": "SQL (Structured Query Language) is a standard language for storing, manipulating, and retrieving data in relational database management systems.", "metadata": "sql_intro"}
        ]
        self.add_documents(sample_docs)
        print(f"✅ Added {len(sample_docs)} sample documents to collection")

    def add_documents(self, documents: List[Dict[str, str]]):
        try:
            contents = [doc["content"] for doc in documents]
            metadatas = [doc.get("metadata", "") for doc in documents]
            embeddings = self.embedding_model.encode(contents).tolist()
            data = [contents, embeddings, metadatas]
            self.collection.insert(data)
            self.collection.flush()
            print(f"✅ Added {len(documents)} documents to vector database")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise e

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            self.collection.load()
            query_embedding = self.embedding_model.encode([query]).tolist()
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "metadata"]
            )
            similar_docs = []
            for hits in results:
                for hit in hits:
                    similar_docs.append({
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    })
            return similar_docs
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        similar_docs = self.search_similar_documents(query, top_k)
        if not similar_docs:
            return ""
        context_parts = []
        for doc in similar_docs:
            context_parts.append(f"Document ({doc['metadata']}): {doc['content']}")
        return "\n\n".join(context_parts)

    @property
    def num_entities(self):
        return self.collection.num_entities

rag_service = None

def get_rag_service() -> Optional[MilvusRAGService]:
    global rag_service
    if rag_service is None:
        try:
            rag_service = MilvusRAGService()
        except Exception as e:
            print(f"⚠️  RAG service not available: {e}")
            return None
    return rag_service 