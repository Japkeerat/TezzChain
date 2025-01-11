import time
import json
import logging
from uuid import uuid4
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator
from logging.handlers import TimedRotatingFileHandler

from tezzchain.chunker import chunker
from tezzchain.llm import llm_providers
from tezzchain.vectordb import vectordb
from tezzchain.core.database import Database
from tezzchain.core.history import ChatHistory
from tezzchain.embedding import embedding_providers
from tezzchain.constants import TEZZCHAIN_LOGGING_DIR
from tezzchain.utilities.hashing import get_file_hash
from tezzchain.utilities.read_file import read_file_as_text
from tezzchain.telemetry.events import SemiAnonymizedTelemetry
from tezzchain.configurations.prepare_configuration import TezzchainConfiguration


class TezzChain:
    def __init__(self, config_file: Optional[Path] = None):
        self.config = TezzchainConfiguration(config_file).get_config()
        with open("temp.json", "w") as f:
            json.dump(self.config, f, indent=4)
        self.logger = self.__configure_logger()
        self.model = self.__set_model()
        self.vectordb = self.__set_vectordb()
        self.embedder = self.__set_embedding_module()
        self.chunking = self.__set_chunking_algorithm()
        self.db = Database()

    def __configure_logger(self):
        logging_levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }
        try:
            logger_level = logging_levels.get(self.config["APP"]["log_level"], "debug")
        except KeyError:
            logger_level = logging_levels["debug"]
        app_status = self.config["APP"].get("app_status", "dev")
        if app_status == "dev":
            logging.basicConfig(
                level=logger_level,
                filename=TEZZCHAIN_LOGGING_DIR
                / f"exec_{datetime.now().strftime('%Y%m%d%H%M%S')}.log",
                filemode="w",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logger = logging.getLogger("tezzchain")
            return logger
        elif "prod":
            logger = logging.getLogger("tezzchain")
            logger.setLevel(logger_level)
            handler = TimedRotatingFileHandler(
                TEZZCHAIN_LOGGING_DIR / "tezzchain.log",
                when="H",
                interval=12,
                backupCount=14,
            )
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            return logger
        else:
            raise ValueError("Invalid APP-STATUS")

    def __set_model(self):
        llm_config = self.config["LLM"]
        provider = self.config["APP"]["llm_provider"].lower()
        return llm_providers[provider](**llm_config)

    def __set_vectordb(self):
        # First set the embedding function as well
        vectordb_config = self.config["VECTORDB"]
        provider = self.config["APP"]["vectordb_provider"].lower()
        return vectordb[provider](**vectordb_config)

    def __set_embedding_module(self):
        embedder_config = self.config["EMBEDDING"]
        provider = self.config["APP"]["embedding_provider"].lower()
        return embedding_providers[provider](**embedder_config)

    def __set_chunking_algorithm(self):
        chunking_config = self.config["CHUNK"]
        algorithm = self.config["APP"]["chunking_algorithm"].lower()
        return chunker[algorithm](**chunking_config)

    def generate(self, query: str, num_predict: Optional[int] = -1) -> Generator:
        embedded_query = self.embedder.embed(query)
        context = self.vectordb.query_db(embedded_query["embeddings"])
        if num_predict >= 0:
            num_predict = num_predict
        else:
            num = self.config["LLM"].get("num_predict", None)
            if num:
                num_predict = num
        # TODO: Integrate prompt template module
        prompt = f"""
        Using the context below, answer the question: {query} \n\n
        Context: {context}
        """
        for chunk in self.model.generate(query=prompt, num_predict=num_predict):
            yield chunk

    def start_a_session(self) -> str:
        return str(uuid4())

    def chat(
        self,
        query: str,
        session: str,
        num_predict: Optional[int] = -1,
    ) -> Generator:
        context = self.vectordb.query_db(self.embedder.embed(query))
        history = ChatHistory(session_id=session)
        history.add_message(context, "context")
        history.add_message(query, "user")
        chats = history.get_messages()
        response = ""
        if num_predict >= 0:
            num_predict = num_predict
        else:
            num = self.config["LLM"].get("num_predict", None)
            if num:
                num_predict = num
        for chunk in self.model.chat(messages=chats, num_predict=num_predict):
            response += chunk["response"]
            chunk["session"] = session
            yield chunk
        history.add_message(response, "assistant")

    def add(
        self, file_path: Path | str, session: str, metadata: Optional[dict] = None
    ) -> str:
        hash = get_file_hash(file_path)
        content = read_file_as_text(file_path)
        chunks = self.chunking.chunk(content)
        if metadata is None:
            metadata = dict()
        metadata["hash"] = hash
        metadata["session"] = session
        for idx, chunk in enumerate(chunks):
            embedding = self.embedder.embed(chunk.text)
            idx = f"{hash}-{idx}"
            self.vectordb.add_content(
                content=chunk.text,
                embedding=embedding["embeddings"][0],
                metadata=metadata,
                id=idx,
            )
        self.db.add_file_to_session(file_hash=hash, session_id=session)
        return hash
