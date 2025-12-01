"""FastAPI entrypoint for the LLM chat service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List, Tuple

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

from app.config import get_settings
from app.database import engine
from app.models import Base
from app.routes import api_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown tasks with database connection retry."""
    import asyncio
    
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)
    
    # Retry database connection with exponential backoff
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting database connection (attempt {attempt}/{max_retries})...")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database connected and tables created successfully")
            break
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"⚠️ Database connection failed (attempt {attempt}/{max_retries}): {e}"
                )
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(
                    f"❌ Database connection failed after {max_retries} attempts: {e}"
                )
                raise
    
    yield
    
    await engine.dispose()
    logger.info("Shutdown complete")



app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router)


def echo_chat(message: str, history: List[Tuple[str, str]] | None) -> str:
    """Return a simple echo response for initial UI testing."""
    _ = history or []
    reply = f"[{settings.default_model_name}] {message}"
    logger.debug("Echoing message via Gradio chat interface: %s", reply)
    return reply


demo = gr.ChatInterface(
    fn=echo_chat,
    title=settings.app_name,
    description="Prototype chat interface. LLM integration coming soon.",
)

mount_gradio_app(app, demo, path=settings.gradio_mount_path)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Default route for convenience."""
    return {"status": "ok", "message": "Visit /docs for the API or /ui for Gradio."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

