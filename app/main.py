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
from openai import AsyncOpenAI

from app.config import (
    AVAILABLE_MODELS,
    DEFAULT_SYSTEM_MESSAGE,
    SYSTEM_MESSAGES,
    get_settings,
)
from app.database import engine
from app.models import Base
from app.routes import api_router

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize OpenRouter client (OpenAI-compatible)
openrouter_client: AsyncOpenAI | None = None

if settings.openrouter_api_key:
    openrouter_client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout=30.0,  # 30 second timeout
    )
    logger.info("OpenRouter client initialized successfully")
else:
    logger.warning("OpenRouter API key not found - chat functionality will be limited")


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


async def chat_with_llm(
    message: str,
    history: List[dict] | List[Tuple[str, str]] | None,
    model_display: str,
    system_preset: str,
) -> str:
    """Chat function that calls OpenRouter API with selected model and system message.
    
    Args:
        message: User's message
        history: Previous conversation history (list of message dicts or tuples)
        model_display: Display name of the selected model
        system_preset: Selected system message preset name
        
    Returns:
        Assistant's response text or error message
    """
    # Check if OpenRouter client is available
    if not openrouter_client:
        error_msg = (
            "❌ OpenRouter API key not configured. "
            "Please set OPENROUTER_API_KEY in your .env file."
        )
        logger.error(error_msg)
        return error_msg
    
    # Get model API identifier from display name
    model_id = AVAILABLE_MODELS.get(model_display)
    if not model_id:
        error_msg = f"❌ Invalid model selected: {model_display}"
        logger.error(error_msg)
        return error_msg
    
    # Get system message from preset
    system_message = SYSTEM_MESSAGES.get(system_preset, DEFAULT_SYSTEM_MESSAGE)
    
    logger.info(
        f"Chat request - Model: {model_display} ({model_id}), "
        f"System Preset: {system_preset}, Message length: {len(message)}"
    )
    
    # Build messages array
    messages = [{"role": "system", "content": system_message}]
    
    # Clean and process conversation history
    # Gradio ChatInterface with type="messages" returns list of dicts with "role" and "content"
    if history:
        cleaned_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
            if isinstance(msg, dict) and "role" in msg and "content" in msg
        ]
        
        # If history is in tuple format (legacy), convert it
        if not cleaned_history and history:
            try:
                # Try to handle tuple format: [(user_msg, assistant_msg), ...]
                for item in history:
                    if isinstance(item, tuple) and len(item) == 2:
                        user_msg, assistant_msg = item
                        cleaned_history.append({"role": "user", "content": user_msg})
                        cleaned_history.append({"role": "assistant", "content": assistant_msg})
            except Exception as e:
                logger.warning(f"Error processing history format: {e}")
        
        # Add cleaned history to messages
        messages.extend(cleaned_history)
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    try:
        # Call OpenRouter API
        logger.debug(f"Sending request to OpenRouter with {len(messages)} messages")
        response = await openrouter_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.7,
        )
        
        # Extract response content
        assistant_response = response.choices[0].message.content
        
        if not assistant_response:
            error_msg = "❌ Received empty response from API"
            logger.warning(error_msg)
            return error_msg
        
        logger.info(f"Successfully received response (length: {len(assistant_response)})")
        return assistant_response
        
    except Exception as e:
        error_msg = f"❌ API Error: {str(e)}"
        logger.error(f"OpenRouter API call failed: {e}", exc_info=True)
        return error_msg


# Global state to store current dropdown values (updated by Gradio events)
_current_model: str = list(AVAILABLE_MODELS.keys())[0]
_current_system: str = "Coding Assistant"


def update_model_selection(model: str) -> str:
    """Update global model selection."""
    global _current_model
    _current_model = model
    logger.info(f"Model selection updated to: {model}")
    return model


def update_system_selection(system: str) -> str:
    """Update global system message selection."""
    global _current_system
    _current_system = system
    logger.info(f"System preset updated to: {system}")
    return system


async def chat_fn(message: str, history: List[dict] | None) -> str:
    """Chat function for Gradio ChatInterface.
    
    Uses global state for model and system message selection.
    Gradio ChatInterface with type="messages" passes history as list of dicts.
    """
    global _current_model, _current_system
    logger.debug(f"Chat called with model={_current_model}, system={_current_system}")
    if history:
        logger.debug(f"History contains {len(history)} messages")
    return await chat_with_llm(message, history, _current_model, _current_system)


# CSS for responsive chat interface height
chat_css = """
    .gradio-container {
        min-height: 70vh !important;
    }
    .chatbot {
        min-height: 600px !important;
        height: 70vh !important;
    }
    #chatbot {
        min-height: 600px !important;
        height: 70vh !important;
    }
"""

# Create Gradio interface with model and system message selection
with gr.Blocks(title=settings.app_name, fill_height=True, css=chat_css) as demo:
    gr.Markdown(f"# {settings.app_name}")
    gr.Markdown("Select your system message preset and model, then start chatting!")
    
    with gr.Row():
        with gr.Column(scale=1):
            system_dropdown = gr.Dropdown(
                choices=list(SYSTEM_MESSAGES.keys()),
                value="Coding Assistant",
                label="System Message Preset",
                info="Choose the assistant's role and behavior",
            )
            # Update global state when dropdown changes
            system_dropdown.change(
                fn=update_system_selection,
                inputs=system_dropdown,
                outputs=system_dropdown,
            )
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=list(AVAILABLE_MODELS.keys())[0],
                label="Model",
                info="Select the LLM model to use",
            )
            # Update global state when dropdown changes
            model_dropdown.change(
                fn=update_model_selection,
                inputs=model_dropdown,
                outputs=model_dropdown,
            )
    
    # Chat interface with responsive height
    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        title="",
        description="",
        type="messages",
    )

mount_gradio_app(app, demo, path=settings.gradio_mount_path)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Default route for convenience."""
    return {"status": "ok", "message": "Visit /docs for the API or /ui for Gradio."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

