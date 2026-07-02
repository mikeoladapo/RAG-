from sqlalchemy.orm import DeclarativeBase,mapped_column,Mapped,relationship
from sqlalchemy.ext.asyncio import async_sessionmaker , create_async_engine
from datetime import datetime,UTC
from sqlalchemy import ForeignKey
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
    file_path: Mapped[str]
    file_type: Mapped[str]
    file_size: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    storage_url : Mapped[str]
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document",cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id : Mapped[int]
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    chunk_index: Mapped[int]
    content: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    document: Mapped["Document"] = relationship(back_populates="chunks")
    