# agents/ingestion_agent.py
"""
IngestionAgent: Processes uploaded documents through OCR-enabled loaders,
chunks them appropriately, and forwards all text chunks to RetrievalAgent.

Supports multiple document formats:
- PDF (with OCR via UnstructuredPDFLoader)
- DOCX, PPTX (via Unstructured loaders)  
- CSV, TXT, MD (via basic loaders)
"""

from __future__ import annotations

import os
import traceback
from typing import List

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.message_broker import MessageBroker
from core.mcp import MCPMessage, IngestRequestPayload


class IngestionAgent:
    """
    Agent responsible for loading, parsing, and chunking documents.
    
    Key Features:
    - Supports multiple file formats with appropriate loaders
    - OCR-enabled PDF processing with fallback
    - Robust error handling per file (one bad file won't stop others)
    - Semantic text chunking with configurable parameters
    - Source attribution for each chunk
    """

    # --------------------------------------------------------------------- #
    # INITIALIZATION
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        broker: MessageBroker,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.broker = broker
        
        # Text splitter for semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            add_start_index=True,  # Helps with chunk metadata
        )
        
        # Document loader mapping by file extension
        self.loader_map = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".pptx": self._load_pptx,
            ".csv": self._load_csv,
            ".txt": self._load_text,
            ".md": self._load_text,
        }
        
        print("IngestionAgent initialized.")

    # --------------------------------------------------------------------- #
    # PUBLIC INTERFACE
    # --------------------------------------------------------------------- #
    def start_listening(self):
        """Start listening for INGEST_REQUEST messages."""
        self.broker.subscribe("ingestion_request_channel", self._handle_ingest_request)
        print("IngestionAgent listening…")

    # --------------------------------------------------------------------- #
    # MESSAGE HANDLING
    # --------------------------------------------------------------------- #
    def _handle_ingest_request(self, message: MCPMessage):
        """Process an INGEST_REQUEST by loading and chunking all files."""
        print(f"IngestionAgent → INGEST_REQUEST (trace_id={message.trace_id})")
        
        try:
            payload = IngestRequestPayload(**message.payload)
            all_chunks = self._process_all_files(payload.file_paths)
            
            if not all_chunks:
                print("⚠️ No chunks produced from any document.")
                return
            
            # Send all chunks to RetrievalAgent in one message
            self._send_chunks_to_retrieval(all_chunks, message.trace_id)
            
        except Exception as exc:
            print(f"❌ Error processing INGEST_REQUEST: {exc}")
            traceback.print_exc()

    # --------------------------------------------------------------------- #
    # FILE PROCESSING
    # --------------------------------------------------------------------- #
    def _process_all_files(self, file_paths: List[str]) -> List[str]:
        """Process all files and return combined text chunks."""
        all_chunks = []
        processed_count = 0
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Skip unsupported file types
            if file_ext not in self.loader_map:
                print(f"⚠️ Unsupported file type: {filename} ({file_ext})")
                continue
            
            try:
                print(f"🔍 Processing: {filename}")
                
                # Load document using appropriate loader
                documents = self.loader_map[file_ext](file_path)
                
                if not documents:
                    print(f"⚠️ No content extracted from {filename}")
                    continue
                
                # Chunk the documents
                chunks = self.text_splitter.split_documents(documents)
                
                # Add source attribution to each chunk
                text_chunks = []
                for chunk in chunks:
                    attributed_content = f"{chunk.page_content}\n[Source: {filename}]"
                    text_chunks.append(attributed_content)
                
                all_chunks.extend(text_chunks)
                processed_count += 1
                print(f"✅ {filename}: {len(text_chunks)} chunks extracted")
                
            except Exception as exc:
                print(f"❌ Failed to process {filename}: {exc}")
                # Continue with other files instead of stopping
                continue
        
        print(f"📦 Processed {processed_count}/{len(file_paths)} files")
        print(f"📦 Total chunks extracted: {len(all_chunks)}")
        return all_chunks

    # --------------------------------------------------------------------- #
    # DOCUMENT LOADERS (by file type)
    # --------------------------------------------------------------------- #
    def _load_pdf(self, file_path: str):
        """Load PDF with OCR support."""
        loader = UnstructuredPDFLoader(
            file_path,
            strategy="hi_res",  # Enables OCR for scanned/image PDFs
            mode="elements",    # Better element-level parsing
        )
        return loader.load()

    def _load_docx(self, file_path: str):
        """Load Word documents."""
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()

    def _load_pptx(self, file_path: str):
        """Load PowerPoint presentations."""
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    def _load_csv(self, file_path: str):
        """Load CSV files."""
        loader = CSVLoader(file_path)
        return loader.load()

    def _load_text(self, file_path: str):
        """Load plain text and markdown files."""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    # --------------------------------------------------------------------- #
    # COMMUNICATION
    # --------------------------------------------------------------------- #
    def _send_chunks_to_retrieval(self, chunks: List[str], trace_id: str):
        """Send processed chunks to RetrievalAgent."""
        build_message = MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            type="BUILD_STORE_REQUEST",
            trace_id=trace_id,
            payload={"text_chunks": chunks},
        )
        
        self.broker.publish("retrieval_channel", build_message)
        print(f"➡️ Sent {len(chunks)} chunks to RetrievalAgent (trace_id={trace_id})")
