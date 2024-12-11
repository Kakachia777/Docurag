from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class RAGEngine:
    """Core RAG (Retrieval Augmented Generation) engine."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_top_k: int = 5,
        cache_dir: Optional[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=cache_dir if cache_dir else ":memory:"
        ))
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create collection for documents
        self.collection = self.vector_store.create_collection(
            name="documentation",
            metadata={"hnsw:space": "cosine"}
        )
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process and index documents for retrieval.
        
        Args:
            documents: List of documents with text and metadata
        """
        for doc in documents:
            # Split text into chunks
            chunks = self.text_splitter.split_text(doc["text"])
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Add to vector store
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=[{**doc["metadata"], "chunk_id": i} for i in range(len(chunks))],
                ids=[f"{doc['metadata']['doc_id']}_{i}" for i in range(len(chunks))]
            )
        
        logger.info(f"Processed and indexed {len(documents)} documents")
    
    def retrieve(self, query: str, filter_criteria: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            filter_criteria: Optional metadata filters
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.similarity_top_k,
            where=filter_criteria
        )
        
        # Format results
        retrieved_chunks = []
        for i in range(len(results["documents"][0])):
            retrieved_chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": float(results["distances"][0][i])
            })
        
        return retrieved_chunks
    
    def update_document(self, doc_id: str, new_text: str, metadata: Dict[str, Any]) -> None:
        """
        Update an existing document in the vector store.
        
        Args:
            doc_id: Document identifier
            new_text: Updated document text
            metadata: Updated metadata
        """
        # Remove existing chunks
        self.collection.delete(
            where={"doc_id": doc_id}
        )
        
        # Process and add new chunks
        self.process_documents([{
            "text": new_text,
            "metadata": {**metadata, "doc_id": doc_id}
        }])
        
        logger.info(f"Updated document {doc_id}")
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document identifier
        """
        self.collection.delete(
            where={"doc_id": doc_id}
        )
        logger.info(f"Deleted document {doc_id}")
    
    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: Document identifier
            top_k: Number of similar documents to retrieve
            
        Returns:
            List of similar documents with similarity scores
        """
        # Get document embeddings
        doc_chunks = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        if not doc_chunks["embeddings"]:
            raise ValueError(f"Document {doc_id} not found")
        
        # Average chunk embeddings for document
        doc_embedding = np.mean(doc_chunks["embeddings"], axis=0)
        
        # Search for similar documents
        results = self.collection.query(
            query_embeddings=[doc_embedding.tolist()],
            n_results=top_k + 1  # Add 1 to account for the query document
        )
        
        # Filter out the query document and format results
        similar_docs = []
        for i in range(len(results["documents"][0])):
            if results["metadatas"][0][i]["doc_id"] != doc_id:
                similar_docs.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": float(results["distances"][0][i])
                })
        
        return similar_docs[:top_k] 