"""PostgreSQL 数据库连接与表管理。"""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None


def _build_url() -> str:
    return (
        f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        logger.info(
            "连接 PostgreSQL %s:%d/%s",
            settings.pg_host,
            settings.pg_port,
            settings.pg_database,
        )
        _engine = create_engine(_build_url())
        _ensure_tables()
    return _engine


def _ensure_tables() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(64) PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                mode VARCHAR(50) DEFAULT 'medical_qa',
                title VARCHAR(255),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(64) REFERENCES sessions(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id, updated_at DESC)
        """))
    logger.info("数据库表初始化完成")


def get_or_create_user(username: str) -> int:
    """获取或创建用户，返回 user_id。"""
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM users WHERE username = :username"),
            {"username": username},
        ).first()
        if row:
            return row[0]
        row = conn.execute(
            text("INSERT INTO users (username) VALUES (:username) RETURNING id"),
            {"username": username},
        ).first()
        logger.info("创建新用户: %s (id=%d)", username, row[0])
        return row[0]


def create_session(session_id: str, user_id: int, mode: str = "medical_qa") -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (id, user_id, mode) "
                "VALUES (:id, :user_id, :mode)"
            ),
            {"id": session_id, "user_id": user_id, "mode": mode},
        )
    logger.info("创建会话: %s (user_id=%d, mode=%s)", session_id, user_id, mode)


def list_user_sessions(user_id: int):
    """返回用户的所有会话，按 updated_at 倒序。"""
    engine = get_engine()
    with engine.connect() as conn:
        return conn.execute(
            text("""
                SELECT s.id, s.mode, s.title, s.created_at, s.updated_at,
                       (SELECT content FROM messages
                        WHERE session_id = s.id AND role = 'human'
                        ORDER BY created_at LIMIT 1
                       ) AS first_question
                FROM sessions s
                WHERE s.user_id = :user_id
                ORDER BY s.updated_at DESC
            """),
            {"user_id": user_id},
        ).fetchall()


def delete_session(session_id: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM sessions WHERE id = :id"),
            {"id": session_id},
        )
    logger.info("删除会话: %s", session_id)


def update_session_mode(session_id: str, mode: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE sessions SET mode = :mode, updated_at = NOW() WHERE id = :id"),
            {"id": session_id, "mode": mode},
        )
