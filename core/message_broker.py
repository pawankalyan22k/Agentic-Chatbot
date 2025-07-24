# core/message_b.py
"""
Robust Redis Pub/Sub message broker for the Agentic RAG system.

Features
--------
•   Automatic (re)connection with exponential back-off
•   Thread-safe, daemonised subscriber loops
•   Graceful handling of socket-closed and connection-lost errors
•   JSON serialisation of MCPMessage objects
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Dict, Any

import redis
from tenacity import retry, stop_after_attempt, wait_exponential

from core.mcp import MCPMessage


class MessageBroker:
    """Production-ready Redis message broker with auto-reconnect."""

    # --------------------------------------------------------------------- #
    # INITIALISATION
    # --------------------------------------------------------------------- #
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db

        self.redis_client: redis.Redis | None = None          # low-level client
        self.pubsub: redis.client.PubSub | None = None        # pubsub object
        self.subscribers: Dict[str, Callable[[MCPMessage], None]] = {}

        self._connect()

    # --------------------------------------------------------------------- #
    # INTERNAL – CONNECTION MANAGEMENT
    # --------------------------------------------------------------------- #
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def _connect(self):
        """Establish a healthy Redis connection (with retry)."""
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
            health_check_interval=30,
            socket_keepalive=True,
        )
        self.redis_client.ping()
        self.pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
        print(f"✅ Connected to Redis @ {self.host}:{self.port}")

    # --------------------------------------------------------------------- #
    # PUBLISH
    # --------------------------------------------------------------------- #
    def publish(self, channel: str, message: MCPMessage):
        """Publish an MCPMessage to a channel (with reconnect on failure)."""
        try:
            self.redis_client.publish(channel, message.model_dump_json())
            print(
                f"📤 Published '{message.type}' on '{channel}' "
                f"(trace_id={message.trace_id})"
            )
        except redis.exceptions.ConnectionError:
            print("🔄 Redis publish failed – reconnecting …")
            self._connect()
            # Retry once after reconnection
            self.redis_client.publish(channel, message.model_dump_json())

    # --------------------------------------------------------------------- #
    # SUBSCRIBE
    # --------------------------------------------------------------------- #
    def subscribe(self, channel: str, callback: Callable[[MCPMessage], None]):
        """Subscribe to a channel; runs callback for every MCPMessage."""

        def _handler(raw: Dict[str, Any]):
            try:
                data = json.loads(raw["data"])
                mcp_msg = MCPMessage(**data)
                print(
                    f"📨 {mcp_msg.type} received on '{channel}' "
                    f"(trace_id={mcp_msg.trace_id})"
                )
                callback(mcp_msg)
            except Exception as exc:  # broad catch to avoid thread kill
                print(f"❌ Error handling message on '{channel}': {exc}")

        # Store and subscribe
        self.subscribers[channel] = _handler
        self.pubsub.subscribe(**{channel: _handler})

        # Background subscriber loop
        def _loop():
            while True:
                try:
                    self.pubsub.get_message(timeout=1)
                except redis.exceptions.ConnectionError:
                    print(f"🔄 Connection lost on '{channel}' – reconnecting …")
                    time.sleep(5)
                    try:
                        self._connect()
                        self.pubsub.subscribe(**{channel: _handler})
                    except Exception as exc:
                        print(f"❌ Re-subscribe failed: {exc}")
                except ValueError as ve:
                    # Raised when socket is closed during shutdown
                    if "closed file" in str(ve):
                        print(f"[INFO] Subscriber '{channel}' shut down cleanly.")
                        break
                    print(f"❌ Subscriber ValueError: {ve}")
                except Exception as exc:
                    print(f"❌ Subscriber error on '{channel}': {exc}")

        threading.Thread(target=_loop, daemon=True).start()
        print(f"🎯 Subscribed to '{channel}'")
