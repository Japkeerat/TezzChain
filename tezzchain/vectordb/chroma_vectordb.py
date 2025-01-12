from pathlib import Path
from typing import Optional

from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection

from tezzchain.constants import TEZZCHAIN_DIR
from tezzchain.vectordb.base import BaseVectorDB


class ChromaDB(BaseVectorDB):
    def __init__(
        self,
        host: Optional[str] = "http://localhost:8000",
        port: Optional[int] = 8000,
        db_path: Optional[Path] = None,
        tenant_id: Optional[str] = "default",
        collection_name: Optional[str] = "default",
        allow_reset: Optional[bool] = False,
        n_results: Optional[int] = 5,
        max_threads: Optional[int] = 40,
    ):
        """
        Use this class to interact with Chroma Vector Database for your RAG application.

        Please note that this will not start ChromaDB server. You must start the server before hand.

        @param host: The host URL at which ChromaDB server is running.
        @param db_path: The path at which the database is present. If not provided, a new folder
        will be created in the home directory.
        """
        self.host = host
        self.port = port
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = TEZZCHAIN_DIR / ".chromadb"
        self.max_threads = max_threads
        self.allow_reset = allow_reset
        self.tenant_id = tenant_id
        self.client = self.__start_client()
        self.collection = self.__create_collection(collection_name)
        self.n_results = n_results

    def __start_client(self) -> ClientAPI:
        settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=self.host,
            chroma_server_http_port=self.port,
            tenant_id=self.tenant_id if self.tenant_id else "default",
            chroma_server_thread_pool_size=self.max_threads,
            allow_reset=self.allow_reset,
        )
        if self.tenant_id:
            client = PersistentClient(
                path=self.db_path, settings=settings, tenant=self.tenant_id
            )
        else:
            client = PersistentClient(path=self.db_path, settings=settings)
        return client

    def __create_collection(self, collection: str) -> Collection:
        collection = self.client.create_collection(name=collection, get_or_create=True)
        return collection

    def get_client(self) -> ClientAPI:
        return self.client

    def get_collection(self) -> Collection:
        return self.collection

    def add_content(self, content: str, embedding: list, metadata: dict, id: str):
        self.collection.add(
            embeddings=[embedding], documents=[content], metadatas=[metadata], ids=[id]
        )

    def query_db(self, query_embedding: list, session_id: Optional[str] = None) -> str:
        if session_id:
            where_clause = {"session": session_id}
        response = self.collection.query(
            query_embedding,
            n_results=self.n_results,
            where=where_clause if session_id else None,
        )
        context = "; ".join(documents[0] for documents in response["documents"])
        return context
