"""Schema migration framework.

Migrations are applied in version order. Each migration is a callable that
receives a SQLAlchemy connection and executes DDL.

Usage:
    from src.migrations import run_migrations
    run_migrations()  # Called from database._ensure_tables()
"""

from sqlalchemy import text

from src.logger import get_logger

logger = get_logger(__name__)

# Migration registry: version → (description, callable)
MIGRATIONS: list[tuple[int, str, callable]] = []


def migration(version: int, description: str):
    """Decorator to register a migration function."""

    def decorator(fn):
        MIGRATIONS.append((version, description, fn))
        return fn

    return decorator


# ── Migrations ──


@migration(1, "Initial schema — users, sessions, messages, chunks, images tables")
def _migration_001(conn) -> None:
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(64) UNIQUE NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    )
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR(64) PRIMARY KEY,
            user_id INT REFERENCES users(id),
            mode VARCHAR(32) DEFAULT 'medical_qa',
            title VARCHAR(128),
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
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    )
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS chunks (
            id VARCHAR(64) PRIMARY KEY,
            content TEXT NOT NULL,
            source VARCHAR(255),
            page INT,
            section_title VARCHAR(512),
            chunk_index INT,
            chunk_type VARCHAR(32) DEFAULT 'text',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    )
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS images (
            id VARCHAR(64) PRIMARY KEY,
            chunk_id VARCHAR(64) REFERENCES chunks(id),
            image_data BYTEA,
            image_format VARCHAR(16),
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
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, created_at)
    """)
    )


@migration(2, "Add schema_version tracking table")
def _migration_002(conn) -> None:
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    )


def run_migrations(engine) -> None:
    """Apply all un-applied migrations in version order."""
    MIGRATIONS.sort(key=lambda m: m[0])

    with engine.begin() as conn:
        # Ensure schema_version table exists (bootstrap)
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        )
        row = conn.execute(text("SELECT COALESCE(MAX(version), 0) FROM schema_version")).scalar()
        current_version = row or 0

        for version, description, fn in MIGRATIONS:
            if version <= current_version:
                continue
            logger.info("Applying migration %d: %s", version, description)
            fn(conn)
            conn.execute(
                text("INSERT INTO schema_version (version) VALUES (:v)"),
                {"v": version},
            )

    logger.info("Migrations complete, version %d", MIGRATIONS[-1][0] if MIGRATIONS else 0)
