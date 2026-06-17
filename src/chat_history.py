"""PostgreSQL 持久化的聊天历史记录。"""

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import text

from src.database import get_engine
from src.logger import get_logger

logger = get_logger(__name__)


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """将聊天历史持久化到 PostgreSQL。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._engine = get_engine()

    @property
    def messages(self) -> list[BaseMessage]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT role, content FROM messages WHERE session_id = :sid ORDER BY created_at"),
                {"sid": self.session_id},
            ).fetchall()

        result: list[BaseMessage] = []
        for role, content in rows:
            if role == "human":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
        return result

    def add_message(self, message: BaseMessage) -> None:
        role = "human" if message.type == "human" else "assistant"
        content = message.content
        if isinstance(content, list):
            content = str(content)

        with self._engine.begin() as conn:
            conn.execute(
                text("INSERT INTO messages (session_id, role, content) VALUES (:sid, :role, :content)"),
                {"sid": self.session_id, "role": role, "content": content},
            )

        # 首条用户消息自动设为会话标题
        if role == "human":
            self._ensure_title(content)

    def clear(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM messages WHERE session_id = :sid"),
                {"sid": self.session_id},
            )
        logger.info("清空会话历史: %s", self.session_id)

    def _ensure_title(self, first_message: str) -> None:
        """仅当会话无标题时，用首条用户消息前 30 字设标题。"""
        title = first_message[:30]
        if len(first_message) > 30:
            title += "..."
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE sessions SET title = :title WHERE id = :sid AND title IS NULL"),
                {"sid": self.session_id, "title": title},
            )
