from uuid import uuid4
from datetime import datetime
from typing import Literal, Optional

from tezzchain.core.database import Database


class ChatHistory:
    def __init__(self, session_id: Optional[str] = None):
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self.__generate_session_id()
        self.history = list()
        self.db = Database()

    def __generate_session_id(self) -> str:
        return str(uuid4())

    def add_message(self, query: str, user_or_assistant: Literal["user", "assistant"]):
        self.db.add_chat_message(
            text=query,
            time=datetime.now(),
            user=user_or_assistant,
            session_id=self.session_id,
        )
        self.history.append({"role": user_or_assistant, "message": query})

    def get_messages(self) -> list[dict[str, str]]:
        messages = self.db.get_chat_history(session_id=self.session_id)
        return [{"role": msg.user, "message": msg.text} for msg in messages]

    def get_session(self):
        return self.session_id
