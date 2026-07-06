from pypdf import PdfReader
from pathlib import Path
from google import genai
import os 
from fastapi import HTTPException,types , Depends
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
    
def chunk(text:str,chunk_size:int=20,overlap:int=2):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks = []
    for start in range (0,len(text),step):
        chunks.append(text[start:start + chunk_size])
    return chunks 

def generate_embedding(text:str) -> list[float]:
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception:
        raise HTTPException(
        status_code=503,
        detail="Failed to generate embedding."
    )

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
    reader = read_file(path)
    chunks = chunk(reader)
    for index , content in enumerate(chunks):
        embeddings = generate_embedding(content)
        db_chunk = Chunk(
            document_id = document.id,
            chunk_index = index,
            content = content ,
            embedding = embeddings
        )
        db.add(db_chunk)
    await db.commit()
    await db.refresh(document)
    return document