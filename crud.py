from fastapi import HTTPException, Path, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Conversation
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pathlib import Path
import shutil
from pypdf import PdfReader

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_file(file:UploadFile) -> Path :
    destination = UPLOAD_DIR / file.filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return destination

def read_file(path:Path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text 

async def create_conversation_service(db: AsyncSession):
    conversation = Conversation(title="New Conversation")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation

async def get_conversations_service(
    db: AsyncSession,
):
    stmt = (
        select(Conversation)
        .order_by(Conversation.created_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_conversation_service(
    conversation_id: int,
    db: AsyncSession,
):
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    result = await db.execute(stmt)
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(
            status_code=404, detail=f"Conversation with id {conversation_id} not found"
        )
    return conversation
