from fastapi import FastAPI,UploadFile,File,Depends,HTTPException
from models import get_db , ChunkResponse,DocumentResponse,Question,Chunk,Document
from sqlalchemy.ext.asyncio import AsyncSession
from services import upload_document_service,generate_chunk_embedding,send_prompt,generate_question_embedding
from sqlalchemy import select

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
    embed_question = generate_question_embedding(question.text)
    cmd = (
        select(Chunk)
        .where(Chunk.document_id == question.document_id)
        .order_by(Chunk.embedding.cosine_distance(embed_question)).limit(10)
    )
    result = await db.execute(cmd)
    chunks = result.scalars().all()
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found."
        )
    context = ""
    for chunk in chunks:
        context += chunk.content + "\n\n"
    prompt = f"""
    You are answering questions about one uploaded document.

    Use ONLY the information contained in the context below.

    If the answer is present, answer it clearly.

    If the answer is not present, reply exactly:

    "I couldn't find that information in the uploaded document."

    Context:
    {context}

    Question:
    {question.text}
    """
    sender = send_prompt(prompt)
    answer = sender.replace("\n\n", " ").replace("\n", " ")
    return {"answer": answer}