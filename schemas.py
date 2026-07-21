from pydantic import BaseModel , ConfigDict
from datetime import datetime,UTC

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

    model_config = ConfigDict(from_attributes=True)

class Question(BaseModel):
    conversation_id: int
    document_id : int 
    text : str 

    model_config = ConfigDict(from_attributes=True)

class ConversationCreate(BaseModel):
    title: str | None = ""


class ConversationUpdate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)