from pypdf import PdfReader
from pathlib import Path
from google import genai
from google.genai import types
import os 
from fastapi import HTTPException, Depends
from pathlib import Path
import shutil
from fastapi import UploadFile,File
from sqlalchemy.ext.asyncio  import AsyncSession 
from models import get_db , Chunk ,Document,Message,Question
from sqlalchemy import select
from rank_bm25 import BM25Okapi
from fastapi.responses import StreamingResponse
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key = api_key)

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
    
def chunk(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 2 <= 1000:
            current += paragraph + "\n\n"
        else:
            chunks.append(current.strip())
            current = paragraph + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks 

def generate_chunk_embedding(chunks:list[str]) -> list[list[float]]:
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunks,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return [embedding.values for embedding in response.embeddings]


    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
def generate_question_embedding(prompt:str):
    try:
        response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=prompt,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
async def stream_prompt(question:str,db:AsyncSession = Depends(get_db),conversation_id:int = None):
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=question,
    )
    full_answer = ""
    try:
        for chunk in response:
            if chunk.text:
                full_answer += chunk.text
                yield chunk.text
        if conversation_id is not None:
            user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=question
        )
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_answer
            )
            db.add_all([user_message, assistant_message])
            await db.commit()
    except Exception:
        await db.rollback()
        raise
async def upload_document_service (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    path = save_file(file)
    document = Document(
        filename=file.filename,
        file_path=str(path),
        file_type=file.content_type,
        file_size=file.size or 0
    )
    try:
        db.add(document)
        await db.flush()
        text = read_file(path)
        chunker = chunk(text)
        chunks = [c.strip() for c in chunker if c.strip()]
        embeddings = generate_chunk_embedding(chunks)
        for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
            db_chunk = Chunk(
                document_id=document.id,
                chunk_index=index,
                content=content,
                embedding=embedding,
            )

            db.add(db_chunk)
        await db.commit()
        await db.refresh(document)
        return document
    except Exception:
        await db.rollback()
        if path.exists():
            path.unlink() 
        raise
    
async def vector_search(query:str,db:AsyncSession,document_id:int,limit:int=10) -> list[Chunk]:
    embed_question = generate_question_embedding(query)
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.embedding.cosine_distance(embed_question)).limit(limit)
    )
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found."
        )
    return chunks 

async def bm25_search(query:str,db:AsyncSession,document_id:int,limit:int=10) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
    )
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found."
        )
    documents = [chunk.content for chunk in chunks]
    tokenized_docs = [document.lower().split() for document in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return [ chunk for score, chunk in ranked[:limit] if score > 0 ]

async def hybrid_search(query:str,db:AsyncSession,document_id:int,limit:int=10) -> list[Chunk]:
    vector_chunks = await vector_search(db=db,document_id=document_id,query=query,limit=limit)
    bm25_chunks = await bm25_search(db=db,document_id=document_id,query=query,limit=limit)
    merged_chunks = {}
    for chunk in vector_chunks:
        merged_chunks[chunk.id] = chunk
    for chunk in bm25_chunks:
        merged_chunks[chunk.id] = chunk
    return list(merged_chunks.values())

async def load_previous_messages(conversation_id:int,db:AsyncSession):
    history = ""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()
    for message in messages:
        role = message.role.capitalize()
        content = message.content
        history += f"{role}: {content}\n"
    return history

async def ask_question_service(question:Question,db:AsyncSession = Depends(get_db)):
    history = await load_previous_messages(conversation_id=question.conversation_id, db=db)
    chunks = await hybrid_search(db=db,document_id=question.document_id,query=question.text)
    context = "\n\n".join(chunk.content for chunk in chunks)
    prompt = f"""
    You are answering questions about one uploaded document.
    Use ONLY the information contained in the context below.
    If the answer is present, answer it clearly.
    If the answer is not present, reply exactly:
    "I couldn't find that information in the uploaded document."
    History:{history}
    Context:{context}
    Question: {question.text} """
    return StreamingResponse(await stream_prompt(prompt, db=db, conversation_id=question.conversation_id,question=question.text),media_type="text/plain")