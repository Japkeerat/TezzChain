from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
)

from tezzchain import constants as const

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, nullable=False)


class SessionAssociatedFiles(Base):
    __tablename__ = "user_uploaded_files"
    file_id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String, nullable=False)
    session_id = Column(Integer, ForeignKey("chat_sessions.session_id"))


class ChatHistory(Base):
    __tablename__ = "chat_history"
    message_id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    time = Column(DateTime, nullable=False)
    user = Column(String, nullable=False)
    session_id = Column(Integer, ForeignKey("chat_sessions.session_id"))


class Database:
    def __init__(self, db_url=f"sqlite:///{const.TEZZCHAIN_DB}"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_chat_session(self, start_time):
        session = self.Session()
        new_session = ChatSession(start_time=start_time)
        session.add(new_session)
        session.commit()
        session.refresh(new_session)
        session.close()
        return new_session.session_id

    def add_file_to_session(self, file_hash, session_id):
        session = self.Session()
        file_id = SessionAssociatedFiles(file_hash=file_hash, session_id=session_id)
        session.add(file_id)
        session.commit()
        session.close()

    def add_chat_message(self, text, time, user, session_id):
        session = self.Session()
        new_message = ChatHistory(
            text=text, time=time, user=user, session_id=session_id
        )
        session.add(new_message)
        session.commit()
        session.close()

    def get_chat_history(self, session_id):
        session = self.Session()
        messages = (
            session.query(ChatHistory)
            .filter_by(session_id=session_id)
            .order_by(ChatHistory.time.asc())
            .all()
        )
        session.close()
        return messages

    def get_all_sessions(self):
        session = self.Session()
        sessions = session.query(ChatSession).all()
        session.close()
        return sessions

    def close(self):
        self.engine.dispose()
