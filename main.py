from fastapi import FastAPI,UploadFile,File,Depends
from models import get_db , Chunk ,Document
from sqlalchemy.ext.asyncio import AsyncSession
from helpers import save_file,read_file,chunk,generate_embedding

app = FastAPI()
@app.post("/documents")
async def upload_document (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    path = save_file(file)
    text = read_file(path)
    chunks = chunk(text)
    e_chunk = generate_embedding(chunks)
    db_document = Document(**file.model_dump(),)