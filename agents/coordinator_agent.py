# agents/coordinator_agent.py
"""
Agent: Orchestrates the entire RAG workflow.

Responsibilities
----------------
•   Receives requests from the UI (ingestion + query)
•   Forwards INGEST_REQUESTS to IngestionAgent and waits for completion
•   Forwards RETRIEVAL_REQUESTS to RetrievalAgent and waits for FINAL_RESPONSE
•   Maintains thread-safe tracking dictionaries for request/response events
•   Publishes system-ready status so the UI can block / unblock input
"""

from __future__ import annotations

import threading
from typing import List, Dict, Any

from core.message_broker import MessageBroker
from core.mcp import (
    MCPMessage,
    IngestRequestPayload,
    RetrievalRequestPayload,
    FinalResponsePayload,
    IngestCompletePayload,
)
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.LLM_response_agent import LLMResponseAgent


class CoordinatorAgent:
    """Central orchestrator for the multi-agent RAG chatbot."""

    # ------------------------------------------------------------------ #
    # INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self):
        self.broker = MessageBroker()

        # Spin-up specialist agents
        self.ingestion_agent = IngestionAgent(self.broker)
        self.retrieval_agent = RetrievalAgent(self.broker)
        self.llm_agent = LLMResponseAgent(self.broker)

        # Async synchronisation primitives
        self._resp_events: Dict[str, threading.Event] = {}
        self._resp_payloads: Dict[str, FinalResponsePayload] = {}
        self._ingest_events: Dict[str, threading.Event] = {}
        self.ingestion_status: Dict[str, str] = {}

        print("CoordinatorAgent initialised.")

    # ------------------------------------------------------------------ #
    # START ALL AGENTS
    # ------------------------------------------------------------------ #
    def start_all_agents(self):
        self.ingestion_agent.start_listening()
        self.retrieval_agent.start_listening()
        self.llm_agent.start_listening()

        # Subscribe for callback messages
        self.broker.subscribe("coordinator_channel", self._handle_callback)
        print("All specialist agents are listening.")

    # ------------------------------------------------------------------ #
    # PUBLIC API CALLED BY UI
    # ------------------------------------------------------------------ #
    def handle_document_ingestion(self, file_paths: List[str]) -> str:
        """Send files for ingestion and return trace_id for UI tracking."""
        payload = IngestRequestPayload(file_paths=file_paths)
        msg = MCPMessage(
            sender="CoordinatorAgent",
            receiver="IngestionAgent",
            type="INGEST_REQUEST",
            payload=payload.model_dump(),
        )
        # Event to wait for INGEST_COMPLETE
        self._ingest_events[msg.trace_id] = threading.Event()

        self.broker.publish("ingestion_request_channel", msg)
        print(f"Coordinator → INGEST_REQUEST (trace_id={msg.trace_id})")
        return msg.trace_id

    def wait_for_ingestion(self, trace_id: str, timeout: float = 180.0) -> bool:
        """Block until ingestion completes (UI uses kb_ready flag)."""
        ev = self._ingest_events.get(trace_id)
        if not ev:
            return False
        done = ev.wait(timeout=timeout)
        self._ingest_events.pop(trace_id, None)
        return done

    def handle_query(self, query: str, timeout: float = 90.0) -> FinalResponsePayload:
        """Forward query to RetrievalAgent and block until FINAL_RESPONSE."""
        req = MCPMessage(
            sender="CoordinatorAgent",
            receiver="RetrievalAgent",
            type="RETRIEVAL_REQUEST",
            payload=RetrievalRequestPayload(query=query).model_dump(),
        )
        # Create event to await FINAL_RESPONSE
        self._resp_events[req.trace_id] = threading.Event()
        self.broker.publish("retrieval_channel", req)
        print(f"Coordinator → RETRIEVAL_REQUEST (trace_id={req.trace_id})")

        event_set = self._resp_events[req.trace_id].wait(timeout=timeout)
        self._resp_events.pop(req.trace_id, None)

        if not event_set:
            return FinalResponsePayload(
                answer="Error: request timed-out – please try again.",
                source_chunks=[],
            )
        return self._resp_payloads.pop(req.trace_id)

    def handle_clear_knowledge_base(self) -> str:
        """Clear FAISS index on disk and reset retrieval agent."""
        if RetrievalAgent.clear_knowledge_base():
            self.retrieval_agent = RetrievalAgent(self.broker)
            self.retrieval_agent.start_listening()
            return "Knowledge base cleared."
        return "Knowledge base already empty."

    # ------------------------------------------------------------------ #
    # INTERNAL CALLBACK HANDLER
    # ------------------------------------------------------------------ #
    def _handle_callback(self, message: MCPMessage):
        """Handle INGEST_COMPLETE and FINAL_RESPONSE messages."""
        if message.type == "INGEST_COMPLETE":
            payload = IngestCompletePayload(**message.payload)
            print(f"Coordinator ⇦ INGEST_COMPLETE (trace_id={message.trace_id})")
            self.ingestion_status[message.trace_id] = payload.message

            ev = self._ingest_events.get(message.trace_id)
            if ev:
                ev.set()

        elif message.type == "FINAL_RESPONSE":
            payload = FinalResponsePayload(**message.payload)
            print(f"Coordinator ⇦ FINAL_RESPONSE (trace_id={message.trace_id})")
            self._resp_payloads[message.trace_id] = payload

            ev = self._resp_events.get(message.trace_id)
            if ev:
                ev.set()

        else:
            print(f"⚠️ Callback with unknown message type: {message.type}")
