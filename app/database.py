"""Database engine and session helpers."""

from __future__ import annotations
from collections.abc import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import get_settings

settings = get_settings()

# Use database_url directly (already has +asyncpg from environment)
engine = create_async_engine(
    settings.database_url,  # ← Changed from async_database_url
    echo=settings.environment == "development",
    future=True,
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    expire_on_commit=False,
)

async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Provide an async database session per request."""
    async with AsyncSessionFactory() as session:
        yield session

