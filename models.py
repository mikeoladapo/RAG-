from sqlalchemy.orm import DeclarativeBase,mapped_column,Mapped,relationship
from sqlalchemy.ext.asyncio import async_sessionmaker , create_async_engine
from datetime import datetime,UTC
from sqlalchemy import ForeignKey ,Text,String
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel,ConfigDict
from dotenv import load_dotenv
import os
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine,expire_on_commit=False)

class Base(DeclarativeBase):
    pass 

class Document(Base):
    __tablename__ = "documents"
    id : Mapped[int] = mapped_column(primary_key=True)
    filename : Mapped[str] 
    file_path: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=lambda:datetime.now(UTC))
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document",cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id : Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(default=lambda:datetime.now(UTC))
    document: Mapped["Document"] = relationship(back_populates="chunks")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session  

class ChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_path: str
    file_type: str
    file_size: int
    created_at: datetime
    chunks: list[ChunkResponse] = []

    model_config = ConfigDict(from_attributes=True)

class Question(BaseModel):
    question : str 
    
    model_config = ConfigDict(from_attributes=True)