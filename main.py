from fastapi import FastAPI,UploadFile,File,Depends,HTTPException
from models import get_db , ChunkResponse,DocumentResponse,Question,Chunk
from sqlalchemy.ext.asyncio import AsyncSession
from services import upload_document_service,generate_embedding,send_prompt
from sqlalchemy import select

app = FastAPI()
@app.post("/documents",response_model=DocumentResponse)
async def upload_document (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    document = await upload_document_service(file, db)
    return document 

@app.post("/ask_question")
async def ask_question(question:Question,db:AsyncSession = Depends(get_db)):
    embed_question = generate_embedding(question.text)
    cmd = (
        select(Chunk)
        .where(Chunk.document_id == question.document_id)
        .order_by(Chunk.embedding.cosine_distance(embed_question)).limit(5)
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
    You are a helpful assistant.

    Answer the question using ONLY the information provided in the context.

    If the answer cannot be found in the context, reply:
    "I couldn't find that information in the uploaded documents."

    Context:
    {context}

    Question:
    {question.text}
    """
    answer = send_prompt(prompt)
    return {"answer": answer}