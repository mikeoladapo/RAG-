from fastapi import FastAPI,UploadFile,File,Depends
from models import get_db , Chunk ,Document
from sqlalchemy.ext.asyncio import AsyncSession
from base.services import save_file,read_file,chunk,generate_embedding,upload_document_service

app = FastAPI()
@app.post("/documents")
async def upload_document (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    document = await upload_document_service(file, db)
    return document 