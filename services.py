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
from models import get_db , Chunk ,Document 

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
        if len(current) + len(paragraph) < 1000:
            current += paragraph + "\n\n"
        else:
            chunks.append(current)
            current = paragraph + "\n\n"
    if current:
        chunks.append(current)
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
    
    return response.text

async def upload_document_service (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    path = save_file(file)
    document = Document(
        filename=file.filename,
        file_path=str(path),
        file_type=file.content_type,
        file_size=file.size or 0
    )
    db.add(document)
    await db.flush()
    text = read_file(path)
    chunks = chunk(text)
    print(f"Text length: {len(text)}")
    print(f"Chunks created: {len(chunks)}")
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