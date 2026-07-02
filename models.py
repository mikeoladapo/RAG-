from sqlalchemy.orm import DeclarativeBase,mapped_column,Mapped,relationship
from sqlalchemy.ext.asyncio import async_sessionmaker , create_async_engine
from datetime import datetime,UTC
from sqlalchemy import ForeignKey ,Text,String
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import os
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)
Session = async_sessionmaker(engine)
session = Session()

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
    