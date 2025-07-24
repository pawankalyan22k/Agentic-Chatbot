# agents/llm_response_agent.py
"""
LLMResponseAgent: Generates final responses using Google's Gemini 2.5 Pro.

Responsibilities:
- Process RETRIEVAL_RESULT messages with query and context
- Generate contextually-aware responses using Gemini 2.5 Pro
- Handle cases where insufficient context is available
- Send FINAL_RESPONSE messages back to CoordinatorAgent
- Comprehensive error handling and performance logging
"""

from __future__ import annotations

import os
import time
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from core.message_broker import MessageBroker
from core.mcp import MCPMessage, FinalResponsePayload


class LLMResponseAgent:
    """
    Agent responsible for generating final responses using Gemini 2.5 Pro.
    
    Key Features:
    - Integration with Google's Gemini 2.5 Pro via LangChain
    - Context-aware response generation with source attribution
    - Robust error handling and timeout management
    - Performance logging and debugging capabilities
    - Graceful handling of insufficient context scenarios
    """

    # --------------------------------------------------------------------- #
    # INITIALIZATION
    # --------------------------------------------------------------------- #
    def __init__(self, broker: MessageBroker):
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")   
        # Validate API key
        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not found in environment variables. "
                "Please create a .env file with your Gemini API key."
            )
        
        self.broker = broker
        
        # Initialize Gemini 2.5 Pro
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.8,  # Low temperature for factual responses
            max_tokens=4096,  # Adequate for detailed responses
            convert_system_message_to_human=True,  # Gemini requirement
        )
        
        # Create optimized prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Build the processing chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
        print("✅ LLMResponseAgent initialized with Gemini 2.5 Pro")

    # --------------------------------------------------------------------- #
    # PUBLIC INTERFACE
    # --------------------------------------------------------------------- #
    def start_listening(self):
        """Start listening for RETRIEVAL_RESULT messages."""
        self.broker.subscribe("llm_channel", self._handle_llm_request)
        print("🎯 LLMResponseAgent listening…")

    # --------------------------------------------------------------------- #
    # MESSAGE HANDLING
    # --------------------------------------------------------------------- #
    def _handle_llm_request(self, message: MCPMessage):
        """Process RETRIEVAL_RESULT message and generate response."""
        start_time = time.time()
        trace_id = message.trace_id
        
        print(f"🧠 Processing LLM request (trace_id={trace_id})")
        
        try:
            # Extract query and context from message
            query = message.payload.get("query", "")
            context_chunks = message.payload.get("context", [])
            
            # Validate input
            if not query:
                final_answer = "Error: No query provided in the request."
            else:
                # Generate response using context
                final_answer = self._generate_contextual_response(query, context_chunks)
            
            # Create response payload
            response_payload = FinalResponsePayload(
                answer=final_answer,
                source_chunks=context_chunks
            )
            
            # Send response back to Coordinator
            final_message = MCPMessage(
                sender="LLMResponseAgent",
                receiver="CoordinatorAgent",
                type="FINAL_RESPONSE",
                trace_id=trace_id,
                payload=response_payload.model_dump()
            )
            
            self.broker.publish("coordinator_channel", final_message)
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            print(f"✅ LLM response sent (trace_id={trace_id}) in {elapsed_time:.2f}s")
            
        except Exception as exc:
            # Handle any errors gracefully
            elapsed_time = time.time() - start_time
            print(f"❌ LLM processing error after {elapsed_time:.2f}s: {exc}")
            
            # Send error response
            error_payload = FinalResponsePayload(
                answer=f"I encountered an error while processing your request: {str(exc)}",
                source_chunks=[]
            )
            
            error_message = MCPMessage(
                sender="LLMResponseAgent",
                receiver="CoordinatorAgent",
                type="FINAL_RESPONSE",
                trace_id=trace_id,
                payload=error_payload.model_dump()
            )
            
            self.broker.publish("coordinator_channel", error_message)

    # --------------------------------------------------------------------- #
    # RESPONSE GENERATION
    # --------------------------------------------------------------------- #
    def _generate_contextual_response(self, query: str, context_chunks: List[str]) -> str:
        """Generate response based on query and retrieved context."""
        
        # Handle empty or error context
        if not context_chunks or self._is_error_context(context_chunks):
            return "I don't have enough information in the provided documents to answer this question."
        
        # Format context for the prompt
        formatted_context = self._format_context(context_chunks)
        
        try:
            # Generate response using the LLM chain
            response = self.chain.invoke({
                "context": formatted_context,
                "query": query
            })
            
            # Clean and return response
            return response.strip()
            
        except Exception as exc:
            print(f"❌ Error during LLM generation: {exc}")
            return f"I encountered an error while generating a response: {str(exc)}"

    def _create_prompt_template(self) -> PromptTemplate:
        """Create optimized prompt template for document Q&A."""
        
        prompt_text = """You are an expert AI assistant specializing in document analysis and question answering.

**CRITICAL INSTRUCTIONS:**
1. Base your answer STRICTLY on the provided context below
2. If the context doesn't contain sufficient information to answer the question, respond with: "I don't have enough information in the provided documents to answer this question."
3. When you do have relevant information, provide a comprehensive, well-structured, and detailed answer
4. Include specific details, examples, and quotes from the context when available
5. Maintain a professional, helpful, and informative tone
6. Do not make up or infer information not present in the context
7. If asked about sources, refer to the source attributions in the context

**CONTEXT:**
{context}

**QUESTION:** {query}

**RESPONSE:**"""

        return PromptTemplate.from_template(prompt_text)

    def _format_context(self, context_chunks: List[str]) -> str:
        """Format context chunks for optimal LLM processing."""
        if not context_chunks:
            return "No context available."
        
        # Join chunks with clear separators
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, 1):
            formatted_chunks.append(f"--- Document Section {i} ---\n{chunk}")
        
        return "\n\n".join(formatted_chunks)

    def _is_error_context(self, context_chunks: List[str]) -> bool:
        """Check if context contains error messages."""
        if len(context_chunks) == 1:
            first_chunk = context_chunks[0].lower()
            return any(error_indicator in first_chunk for error_indicator in [
                "error:", "empty", "no documents", "knowledge base is empty"
            ])
        return False

    # --------------------------------------------------------------------- #
    # UTILITY METHODS
    # --------------------------------------------------------------------- #
    def get_model_info(self) -> dict:
        """Get information about the current LLM configuration."""
        return {
            "model": "gemini-2.5-pro",
            "provider": "Google",
            "temperature": 0.1,
            "max_tokens": 4096,
            "status": "ready"
        }

    def health_check(self) -> bool:
        """Perform a simple health check of the LLM connection."""
        try:
            # Simple test query
            test_response = self.chain.invoke({
                "context": "This is a test context.",
                "query": "What is this?"
            })
            return bool(test_response and len(test_response.strip()) > 0)
        except Exception as exc:
            print(f"❌ LLM health check failed: {exc}")
            return False
