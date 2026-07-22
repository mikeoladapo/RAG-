from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from models import Conversation, Message,Chunk,get_db
import os
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi
from sqlalchemy import select

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key = api_key)

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
    
async def stream_prompt(prompt:str,query:str,db:AsyncSession = Depends(get_db),conversation_id:int = None):
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=query,
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
            content=query
        )
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_answer
            )
            db.add_all([user_message, assistant_message])
            conversation = await db.get(
            Conversation,
            conversation_id,
            )
            if conversation.title == "New Conversation":
                conversation.title = await generate_conversation_title(query)
            await db.commit()
    except Exception:
        await db.rollback()
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

async def generate_conversation_title(question: str) -> str:
    prompt = f"""Generate a short conversation title (3-6 words) for this question.
    Question:
    {question}
    Return ONLY the title."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip() 