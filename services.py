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
from models import get_db , Chunk ,Document,DocumentResponse,Question
from sqlalchemy import select
from rank_bm25 import BM25Okapi

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
def send_prompt(prompt:str):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.replace("\\n", "\n")

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