from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import start_http_server

from rag_engine.core import RAGEngine
from llm.generator import ResponseGenerator
from utils.cache import RedisCache
from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="User query")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

class QueryResponse(BaseModel):
    """Query response model."""
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    followup_questions: List[str] = Field(..., description="Suggested follow-up questions")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DocumentRequest(BaseModel):
    """Document indexing request model."""
    text: str = Field(..., description="Document text")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")

class RAGService:
    """RAG service with API endpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.rag_engine = RAGEngine(
            embedding_model=config["embedding"]["model"],
            chunk_size=config["chunking"]["size"],
            chunk_overlap=config["chunking"]["overlap"],
            cache_dir=config["storage"]["cache_dir"]
        )
        
        self.generator = ResponseGenerator(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"]
        )
        
        self.cache = RedisCache(
            host=config["redis"]["host"],
            port=config["redis"]["port"],
            ttl=config["redis"]["ttl"]
        )
        
        self.rate_limiter = RateLimiter(
            requests_per_minute=config["security"]["rate_limit"]["requests_per_minute"],
            burst=config["security"]["rate_limit"]["burst"]
        )
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="DocuRAG API",
            description="RAG-powered documentation assistant API",
            version="1.0.0"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Start metrics server
        start_http_server(config["monitoring"]["prometheus_port"])
    
    def _setup_middleware(self):
        """Setup API middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["security"]["cors"]["allowed_origins"],
            allow_methods=self.config["security"]["cors"]["allowed_methods"],
            allow_headers=["*"]
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        api_key_header = APIKeyHeader(name="X-API-Key")
        
        @self.app.post("/api/v1/query", response_model=QueryResponse)
        async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Security(api_key_header)
        ) -> QueryResponse:
            # Validate API key
            if api_key not in self.config["security"]["api_keys"]:
                raise HTTPException(status_code=403, detail="Invalid API key")
            
            # Apply rate limiting
            if not self.rate_limiter.allow_request(api_key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Check cache
            cache_key = f"query:{request.query}:{request.conversation_id}"
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return QueryResponse(**cached_response)
            
            try:
                start_time = datetime.utcnow()
                
                # Retrieve relevant chunks
                retrieved_chunks = self.rag_engine.retrieve(
                    query=request.query,
                    filter_criteria=request.metadata_filters
                )
                
                # Get chat history if conversation_id provided
                chat_history = None
                if request.conversation_id:
                    chat_history = self.cache.get(f"chat:{request.conversation_id}")
                
                # Generate response
                response_data = self.generator.generate_response(
                    query=request.query,
                    retrieved_chunks=retrieved_chunks,
                    chat_history=chat_history
                )
                
                # Generate follow-up questions in background
                background_tasks.add_task(
                    self.generator.generate_followup_questions,
                    request.query,
                    response_data["response"],
                    retrieved_chunks
                )
                
                # Prepare response
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                response = QueryResponse(
                    response=response_data["response"],
                    sources=[{
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "similarity": chunk["similarity"]
                    } for chunk in retrieved_chunks],
                    followup_questions=[],  # Will be updated in background
                    conversation_id=request.conversation_id,
                    processing_time=processing_time,
                    timestamp=datetime.utcnow()
                )
                
                # Cache response
                self.cache.set(cache_key, response.dict())
                
                # Update chat history
                if request.conversation_id:
                    self._update_chat_history(
                        request.conversation_id,
                        request.query,
                        response_data["response"]
                    )
                
                return response
            
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                raise HTTPException(status_code=500, detail="Query processing failed")
        
        @self.app.post("/api/v1/documents")
        async def index_document(
            request: DocumentRequest,
            api_key: str = Security(api_key_header)
        ):
            # Validate API key
            if api_key not in self.config["security"]["api_keys"]:
                raise HTTPException(status_code=403, detail="Invalid API key")
            
            try:
                self.rag_engine.process_documents([{
                    "text": request.text,
                    "metadata": request.metadata
                }])
                
                return {"status": "success", "message": "Document indexed successfully"}
            
            except Exception as e:
                logger.error(f"Error indexing document: {str(e)}")
                raise HTTPException(status_code=500, detail="Document indexing failed")
    
    def _update_chat_history(
        self,
        conversation_id: str,
        query: str,
        response: str
    ) -> None:
        """Update chat history in cache."""
        history_key = f"chat:{conversation_id}"
        history = self.cache.get(history_key) or []
        
        history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])
        
        # Keep only last 10 messages
        history = history[-10:]
        self.cache.set(history_key, history)

def create_app(config: Dict[str, Any]) -> FastAPI:
    """Create and configure the FastAPI application."""
    service = RAGService(config)
    return service.app 