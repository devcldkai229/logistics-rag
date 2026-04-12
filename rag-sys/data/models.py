import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class FeedbackChunk(Base):
    __tablename__ = "feedback_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    feedback_id: Mapped[str | None] = mapped_column(
        String(50),
        index=True,
        nullable=True,
    )
    order_code: Mapped[str | None] = mapped_column(
        String(50),
        index=True,
        nullable=True,
    )
    customer_code: Mapped[str] = mapped_column(String(50), index=True)
    customer_tier: Mapped[str] = mapped_column(String(50))
    route_name: Mapped[str] = mapped_column(String(100))
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentiment: Mapped[str] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        insert_default=dict,
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_name: Mapped[str] = mapped_column(String(255), index=True)
    section_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        insert_default=dict,
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
