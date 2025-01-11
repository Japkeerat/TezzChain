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
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = "default",
        allow_reset: Optional[bool] = False,
        n_results: Optional[int] = 5,
    ):
        """
        Use this class to interact with Chroma Vector Database for your RAG application.

        Please note that this will not start ChromaDB server. You must start the server before hand.

        @param host: The host URL at which ChromaDB server is running.
        @param db_path: The path at which the database is present. If not provided, a new folder
        will be created in the home directory.
        """
        self.host = host
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = TEZZCHAIN_DIR / ".chromadb"
        self.client = self.__start_client()
        self.allow_reset = allow_reset
        self.collection = self.__create_collection(collection_name)
        self.n_results = n_results

    def __start_client(self) -> ClientAPI:
        settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host="http://localhost:8000/",
            chroma_server_http_port=8000,
            # allow_reset=self.allow_reset,
        )
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
            pass
        response = self.collection.query(query_embedding, n_results=self.n_results)
        print(response["documents"])
        context = "; ".join(documents[0] for documents in response["documents"])
        return context
