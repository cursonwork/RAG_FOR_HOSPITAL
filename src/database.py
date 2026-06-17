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

    # Apply any pending migrations (for future schema changes)
    from src.migrations import run_migrations

    run_migrations(engine)

    # Legacy inline DDL — ensures backward compatibility for existing installs
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(64) PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                mode VARCHAR(50) DEFAULT 'medical_qa',
                title VARCHAR(255),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(64) REFERENCES sessions(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at)
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id, updated_at DESC)
        """)
        )

        # ── 知识库分块与图片表 ──
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR(64) PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255) NOT NULL,
                page INT DEFAULT 0,
                section_title VARCHAR(512),
                chunk_index INT DEFAULT 0,
                chunk_type VARCHAR(20) DEFAULT 'text',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS images (
                id VARCHAR(64) PRIMARY KEY,
                chunk_id VARCHAR(64) REFERENCES chunks(id),
                image_data BYTEA NOT NULL,
                image_format VARCHAR(10) DEFAULT 'png',
                description TEXT,
                caption TEXT,
                source VARCHAR(255),
                page INT,
                bbox_x0 FLOAT,
                bbox_y0 FLOAT,
                bbox_x1 FLOAT,
                bbox_y1 FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source
            ON chunks(source, page)
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS idx_images_source
            ON images(source, page)
        """)
        )

        # 为已有 images 表补充 bbox 列（兼容旧表）
        _ALLOWED_COLS = {"bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"}
        for col in _ALLOWED_COLS:
            conn.execute(text(f"ALTER TABLE images ADD COLUMN IF NOT EXISTS {col} FLOAT"))
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
            text("INSERT INTO sessions (id, user_id, mode) VALUES (:id, :user_id, :mode)"),
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


# ── 知识库分块 CRUD ──


def save_chunk(
    chunk_id: str,
    content: str,
    source: str,
    page: int = 0,
    section_title: str = "",
    chunk_index: int = 0,
    chunk_type: str = "text",
) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO chunks (id, content, source, page, section_title, chunk_index, chunk_type) "
                "VALUES (:id, :content, :source, :page, :section_title, :chunk_index, :chunk_type)"
            ),
            {
                "id": chunk_id,
                "content": content,
                "source": source,
                "page": page,
                "section_title": section_title,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
            },
        )


def get_chunk(chunk_id: str) -> dict | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, content, source, page, section_title, chunk_index, chunk_type FROM chunks WHERE id = :id"),
            {"id": chunk_id},
        ).first()
    if row is None:
        return None
    return dict(row._mapping)


def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, content, source, page, section_title, chunk_index, chunk_type FROM chunks "
                "WHERE id = ANY(:ids)"
            ),
            {"ids": chunk_ids},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def count_chunks() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM chunks")).first()
    return row[0]


def clear_all_chunks() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM images"))
        conn.execute(text("DELETE FROM chunks"))
    logger.info("已清空 chunks 和 images 表")


def update_chunk_content(chunk_id: str, content: str) -> None:
    """更新已有 chunk 的文本内容，用于图片描述从占位更新为实际描述。"""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE chunks SET content = :content WHERE id = :id"),
            {"id": chunk_id, "content": content},
        )


# ── 图片 CRUD ──


def save_image(
    image_id: str,
    chunk_id: str,
    image_data: bytes,
    description: str = "",
    caption: str = "",
    source: str = "",
    page: int = 0,
    image_format: str = "png",
    bbox: tuple | None = None,
) -> None:
    engine = get_engine()
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox if bbox else (None, None, None, None)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO images (id, chunk_id, image_data, image_format, description, caption, "
                "source, page, bbox_x0, bbox_y0, bbox_x1, bbox_y1) "
                "VALUES (:id, :chunk_id, :image_data, :image_format, :description, :caption, "
                ":source, :page, :bbox_x0, :bbox_y0, :bbox_x1, :bbox_y1)"
            ),
            {
                "id": image_id,
                "chunk_id": chunk_id,
                "image_data": image_data,
                "image_format": image_format,
                "description": description,
                "caption": caption,
                "source": source,
                "page": page,
                "bbox_x0": bbox_x0,
                "bbox_y0": bbox_y0,
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
            },
        )


def update_image_description(image_id: str, description: str) -> None:
    """更新已有图片记录的描述字段。"""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE images SET description = :desc WHERE id = :id"),
            {"id": image_id, "desc": description},
        )


def get_image(image_id: str) -> dict | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, chunk_id, image_data, image_format, description, caption, "
                "source, page, bbox_x0, bbox_y0, bbox_x1, bbox_y1 FROM images WHERE id = :id"
            ),
            {"id": image_id},
        ).first()
    if row is None:
        return None
    result = dict(row._mapping)
    if isinstance(result.get("image_data"), memoryview):
        result["image_data"] = bytes(result["image_data"])
    return result
