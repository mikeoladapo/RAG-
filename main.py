from fastapi import FastAPI,UploadFile,File,Depends
from models import get_db , DocumentResponse,Question,Document,Message
from sqlalchemy.ext.asyncio import AsyncSession
from services import upload_document_service,ask_question_service
from sqlalchemy import select
from schemas import DocumentResponse,ConversationResponse
from crud import create_conversation_service,get_conversations_service,get_conversation_service

app = FastAPI()
@app.post("/upload_document",response_model=DocumentResponse)
async def upload_document (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    document = await upload_document_service(file, db)
    return document 

@app.get("/documents",response_model=list[DocumentResponse])
async def get_documents(db:AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document))
    documents = result.scalars().all()
    return documents 

@app.post("/ask_question")
async def ask_question(question:Question,db:AsyncSession = Depends(get_db)):
    return await ask_question_service(
        question=question,
        db=db,
    )

@app.post("/conversations",response_model=ConversationResponse,status_code=201,)
async def create_conversation(db: AsyncSession = Depends(get_db)):
    return await create_conversation_service(db)

@app.get("/conversations",response_model=list[ConversationResponse],)
async def get_conversations(db: AsyncSession = Depends(get_db),):
    return await get_conversations_service(db)

@app.get("/conversations/{conversation_id}",response_model=ConversationResponse,)
async def get_conversation(conversation_id: int,db: AsyncSession = Depends(get_db)):
    return await get_conversation_service(conversation_id,db,)