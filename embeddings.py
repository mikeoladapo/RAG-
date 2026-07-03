from pypdf import PdfReader
from pathlib import Path
from google import genai
import os 
from fastapi import HTTPException,types


api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key = api_key)

def read_file(path:Path):
    read = PdfReader(path)
    text = ""
    for page in read.pages:
        text += page.extract_text() + "\n"
    return text 
    
def chunk(text:str,count:int=20,overlap:int=2):
    step = count - overlap
    chunks = []
    for start in range (0,len(text),step):
        chunks.append(text[start:start + count])
    return chunks 

def generate_embeddings(text:chunk) -> list[float]:
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