from fastapi import FastAPI,UploadFile,File,Depends,HTTPException
from models import get_db , ChunkResponse,DocumentResponse,Question,Chunk,Document
from sqlalchemy.ext.asyncio import AsyncSession
from services import upload_document_service,generate_chunk_embedding,generate_question_embedding,vector_search,hybrid_search,stream_prompt
from sqlalchemy import select
from fastapi.responses import StreamingResponse

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
    chunks = await hybrid_search(db=db,document_id=question.document_id,query=question.text)
    context = "\n\n".join(chunk.content for chunk in chunks)
    prompt = f"""
    You are answering questions about one uploaded document.
    Use ONLY the information contained in the context below.
    If the answer is present, answer it clearly.
    If the answer is not present, reply exactly:
    "I couldn't find that information in the uploaded document."
    Context:{context}
    Question: {question.text} """
    return StreamingResponse(stream_prompt(prompt),media_type="text/plain")