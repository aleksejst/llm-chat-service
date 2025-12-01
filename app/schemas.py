"""Pydantic schemas for the API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    environment: str = Field(default="development")


class UserBase(BaseModel):
    email: EmailStr


class UserRead(UserBase):
    id: int
    created_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class ChatSessionBase(BaseModel):
    title: Optional[str] = None


class ChatSessionRead(ChatSessionBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageBase(BaseModel):
    role: str
    content: str
    llm_model: str


class MessageRead(MessageBase):
    id: int
    session_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class APIUsageRead(BaseModel):
    id: int
    user_id: Optional[int]
    session_id: Optional[int]
    llm_model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Decimal
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatHistory(BaseModel):
    messages: List[MessageRead] = Field(default_factory=list)

