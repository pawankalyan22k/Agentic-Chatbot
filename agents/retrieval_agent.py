# agents/retrieval_agent.py
"""
RetrievalAgent: Manages FAISS vector store for semantic document retrieval.

Responsibilities:
- Embed text chunks using HuggingFace embeddings
- Build and maintain FAISS vector index with persistence
- Perform semantic similarity search for user queries
- Handle vector store lifecycle (create, update, clear)
"""

from __future__ import annotations

import os
import shutil
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from core.message_broker import MessageBroker
from core.mcp import MCPMessage, RetrievalRequestPayload

# Constants
VECTOR_STORE_PATH = "vector_store"


class RetrievalAgent:
    """
    Agent for embedding text chunks, storing them in FAISS, and performing
    fast semantic retrieval for user queries.
    
    Key Features:
    - HuggingFace embeddings with local CPU execution
    - FAISS vector store with disk persistence
    - Incremental vector store updates (add to existing)
    - Robust error handling and logging
    - Configurable similarity search parameters
    """

    # --------------------------------------------------------------------- #
    # INITIALIZATION
    # --------------------------------------------------------------------- #
    def __init__(self, broker: MessageBroker, embedding_model: str = "all-MiniLM-L6-v2"):
        self.broker = broker
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,# Ensures CPU compatibility
            show_progress=False,  # Reduces console noise
        )
        
        # Load existing vector store or initialize as None
        self.vector_store: Optional[FAISS] = self._load_vector_store()
        
        print("RetrievalAgent initialized.")

    # --------------------------------------------------------------------- #
    # PUBLIC INTERFACE
    # --------------------------------------------------------------------- #
    def start_listening(self):
        """Start listening for retrieval and build store requests."""
        self.broker.subscribe("retrieval_channel", self._handle_message)
        print("RetrievalAgent listening…")

    # --------------------------------------------------------------------- #
    # MESSAGE HANDLING
    # --------------------------------------------------------------------- #
    def _handle_message(self, message: MCPMessage):
        """Route incoming messages to appropriate handlers."""
        
        if message.type == "BUILD_STORE_REQUEST":
            self._handle_build_store_request(message)
            
        elif message.type == "RETRIEVAL_REQUEST":
            self._handle_retrieval_request(message)
            
        else:
            print(f"⚠️ Unknown message type: {message.type}")

    def _handle_build_store_request(self, message: MCPMessage):
        """Process BUILD_STORE_REQUEST by adding chunks to vector store."""
        print(f"RetrievalAgent → BUILD_STORE_REQUEST (trace_id={message.trace_id})")
        
        try:
            chunks = message.payload.get("text_chunks", [])
            added_count = self._build_or_update_store(chunks)
            
            # Send acknowledgment back to Coordinator
            ack_message = MCPMessage(
                sender="RetrievalAgent",
                receiver="CoordinatorAgent", 
                type="INGEST_COMPLETE",
                trace_id=message.trace_id,
                payload={
                    "status": "Success",
                    "message": f"Knowledge base updated with {added_count} new chunks."
                }
            )
            
            self.broker.publish("coordinator_channel", ack_message)
            print(f"✅ Sent INGEST_COMPLETE confirmation (trace_id={message.trace_id})")
            
        except Exception as exc:
            print(f"❌ Error in BUILD_STORE_REQUEST: {exc}")

    def _handle_retrieval_request(self, message: MCPMessage):
        """Process RETRIEVAL_REQUEST by performing semantic search."""
        print(f"RetrievalAgent → RETRIEVAL_REQUEST (trace_id={message.trace_id})")
        
        try:
            payload = RetrievalRequestPayload(**message.payload)
            retrieved_chunks = self._retrieve_chunks(payload.query)
            
            print(f"[Debug] Retrieved {len(retrieved_chunks)} chunks for query: '{payload.query}'")
            
            # Send results to LLMResponseAgent
            response_message = MCPMessage(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                type="RETRIEVAL_RESULT", 
                trace_id=message.trace_id,
                payload={
                    "query": payload.query,
                    "context": retrieved_chunks
                }
            )
            
            self.broker.publish("llm_channel", response_message)
            print(f"➡️ Sent context to LLMResponseAgent (trace_id={message.trace_id})")
            
        except Exception as exc:
            print(f"❌ Error in RETRIEVAL_REQUEST: {exc}")

    # --------------------------------------------------------------------- #
    # VECTOR STORE MANAGEMENT
    # --------------------------------------------------------------------- #
    def _load_vector_store(self) -> Optional[FAISS]:
        """Load existing FAISS vector store from disk if available."""
        if not os.path.exists(VECTOR_STORE_PATH):
            print("🔄 No existing vector store found - will create new one on first ingestion")
            return None
            
        try:
            print("🔄 Loading existing vector store from disk...")
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                self.embedding_model,
                allow_dangerous_deserialization=True  # Required for FAISS loading
            )
            
            # Log vector count for debugging
            vector_count = vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0
            print(f"✅ Loaded vector store with {vector_count} existing vectors")
            return vector_store
            
        except Exception as exc:
            print(f"❌ Could not load vector store: {exc}")
            print("🔄 Will create new vector store on next ingestion")
            return None

    def _build_or_update_store(self, text_chunks: List[str]) -> int:
        """Create new vector store or add chunks to existing one."""
        if not text_chunks:
            print("⚠️ No text chunks provided - skipping store update")
            return 0
            
        chunk_count = len(text_chunks)
        
        try:
            if self.vector_store is None:
                # Create new vector store
                print(f"🔨 Creating new vector store with {chunk_count} chunks...")
                self.vector_store = FAISS.from_texts(
                    texts=text_chunks,
                    embedding=self.embedding_model
                )
            else:
                # Add to existing vector store
                print(f"➕ Adding {chunk_count} chunks to existing vector store...")
                self.vector_store.add_texts(text_chunks)
            
            # Always save after updates
            print("💾 Saving vector store to disk...")
            self.vector_store.save_local(VECTOR_STORE_PATH)
            
            # Debug logging
            total_vectors = self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
            print(f"✅ Vector store updated successfully")
            print(f"[Debug] Total vectors in store: {total_vectors}")
            
            return chunk_count
            
        except Exception as exc:
            print(f"❌ Error updating vector store: {exc}")
            return 0

    def _retrieve_chunks(self, query: str, k: int = 5) -> List[str]:
        """Perform semantic similarity search and return top-k results."""
        if self.vector_store is None:
            return ["Error: The knowledge base is empty. Please upload and process documents first."]
        
        try:
            print(f"🔍 Searching for top {k} chunks matching: '{query}'")
            
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            # Extract text content from results
            chunks = [doc.page_content for doc in results]
            
            print(f"✅ Found {len(chunks)} relevant chunks")
            return chunks
            
        except Exception as exc:
            print(f"❌ Error during retrieval: {exc}")
            return [f"Error during retrieval: {str(exc)}"]

    # --------------------------------------------------------------------- #
    # UTILITY METHODS
    # --------------------------------------------------------------------- #
    @staticmethod
    def clear_knowledge_base() -> bool:
        """Delete the persisted vector store from disk."""
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                shutil.rmtree(VECTOR_STORE_PATH)
                print("🗑️ Knowledge base cleared from disk")
                return True
            except Exception as exc:
                print(f"❌ Failed to clear knowledge base: {exc}")
                return False
        else:
            print("ℹ️ Knowledge base was already empty")
            return False

    def get_store_info(self) -> dict:
        """Get information about current vector store state."""
        if self.vector_store is None:
            return {
                "status": "empty",
                "vector_count": 0,
                "embedding_dim": 0
            }
        
        try:
            vector_count = self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
            embedding_dim = self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else 0
            
            return {
                "status": "ready",
                "vector_count": vector_count,
                "embedding_dim": embedding_dim
            }
        except:
            return {
                "status": "error",
                "vector_count": 0,
                "embedding_dim": 0
            }
